import autogen
from .agents import ResearcherAgent
import arxiv
import os
import numpy as np

TEST_DATA_PATH = "/Users/autumn/Documents/GitHub/hathor/test_data"
EX_CODE_PATH = "/Users/autumn/Documents/GitHub/hathor/ex_code"
PAPERS_PER_QUERY = 10

class Hathor:
    def __init__(self, config_list=None, prompt=None, data_path=TEST_DATA_PATH, ex_code_path=EX_CODE_PATH, interactive=False, literature=None):
        self.config_list = config_list
        self.prompt = prompt
        self.data_path = data_path
        self.interactive = interactive
        self.ex_code_path = ex_code_path
        self.literature = literature

        self.llm_config = {
            "config_list": self.config_list,
            "temperature": 0.1,
        }

        self._setup_groupchat()

    def run(self):

        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
         
        self.user_proxy.initiate_chat(
            self.hathor,
            message="""
                        Please generate:
                            * a high quality galaxy formation hypothesis that can be evaluated using RAMSES simulation data.
                            * a high quality plot idea that will use RAMSES data to evaluate that hypothesis
                            * thorough, efficient code for creating that plot
                    """
        )
        
        return self.groupchat.messages

    def respond(self, response):
        pass
        
    def _setup_groupchat(self):
        self.hypothesis_brainstormer = self._create_hypoth_brainstormer()
        self.plot_brainstormer = self._create_plot_brainstormer()
        self.critic = self._create_critic()
        # self.hpc_expert = self._create_hpc_expert()
        self.coder = self._create_coder()
        self.executor = self._create_executor()

        self.agents = [self.hypothesis_brainstormer, self.plot_brainstormer, self.critic, self.coder, self.executor]

        allowed_transitions = {
            self.hypothesis_brainstormer: [self.critic],
            self.plot_brainstormer: [self.critic],
            self.critic: [self.hypothesis_brainstormer, self.plot_brainstormer],
            self.coder: [self.executor],
            self.executor: [self.coder]
        }

        def custom_speaker_selection(last_speaker, groupchat):
            # Get the last message
            last_message = groupchat.messages[-1] if groupchat.messages else None

            if len(groupchat.messages) > 3 and len(groupchat.messages) % 3 == 0:
                print(f"Summarizing conversation at {len(groupchat.messages)} messages...")
                
                # Create a detailed summary prompt
                summary_prompt = f"""
                You are summarizing a brainstorming session. Based on the recent conversation, provide:
                
                1. Current hypothesis count and which hypotheses remain (list by number/title)
                2. What percentage was removed in the last critique and why
                3. What the next step should be (more refinement or move to coding?)
                
                Format as a brief structured summary (max 300 words). This summary will be used to continue the conversation.
                
                Current round: {len(groupchat.messages)}
                """
                
                # Create message for summary generation
                summary_request = {
                    "role": "user",
                    "content": summary_prompt,
                    "name": "system"
                }
                
                # Get last 10 messages for context (not all messages, to save tokens)
                recent_context = groupchat.messages[-10:]
                
                try:
                    # Have the critic generate a summary
                    critic_response = self.critic.generate_reply(
                        messages=recent_context + [summary_request],
                        sender=None
                    )
                    
                    # Create a system message with the summary (not from Critic, from "system")
                    summary_msg = {
                        "role": "system",  # Use "system" not "assistant"
                        "content": f"[CONVERSATION SUMMARY - Round {len(groupchat.messages)}]\n\n{critic_response}",
                        "name": "ConversationSummary"  # Give it a distinct name
                    }
                    
                    # Replace old messages: summary + last 3 messages (not 2, gives more context)
                    groupchat.messages = [summary_msg] + groupchat.messages[-3:]
                    
                    print(f"✓ Summarized {len(groupchat.messages)} messages into compact history")
                    
                except Exception as e:
                    print(f"Warning: Summary generation failed: {e}")
                    # Fallback: just trim to last 8 messages without summarizing
                    groupchat.messages = groupchat.messages[-8:]
                    print(f"✓ Fallback: Trimmed to last {len(groupchat.messages)} messages")
            
            # Check if critic said "TERMINATE BRAINSTORMING"
            if last_speaker == self.critic and last_message:
                if "TERMINATE BRAINSTORM" in last_message.get("content", ""):
                    groupchat.messages = groupchat.messages[-2:]

                    return self.coder
        
            # Otherwise, use default transition logic
            return "auto"  

        self.groupchat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=40,
            speaker_selection_method=custom_speaker_selection,
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed"
        )
        self.hathor_llm_config = {
            "config_list": self.config_list,
            "temperature": 0.1,
        }

        self.hathor = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.hathor_llm_config)


    def _get_data_string(self):
        files = sorted(os.listdir(EX_CODE_PATH))
        data_str = ""
        for filename in files:
            filepath = os.path.join(EX_CODE_PATH, filename)
            data = np.fromfile(filepath, count=5)

            # Capture numpy's pretty print as string
            data_str += f"STORED WITHIN {filepath}"
            with np.printoptions(precision=3, suppress=True):
                data_str += np.array2string(data, separator=', ')
            data_str += ("\n" + "="*70)
        return data_str

    def _create_coder(self):
        data_str = self._get_data_string()

        system_message = f"""
                            You are a Python expert. Write clean, efficient code. Make sure to include ledgible comments and docstrings.
                            You are also an expert on the RAMSES simulations. 
                            
                            The user made the following clarifications/requests:
                            {self.prompt}

                            All data can be found in {self.data_path}

                            The first 5 rows of each file in that path can be found here: {data_str}
                            
                            You have access to the following packages: yt, numpy, pandas, matplotlib, scipy

                            When possible, please use tools from the open source package known as yt. 
                            
                            Below is a summary of user-provided example code. Please use this as guidance.
                            {self._get_all_files_str(self.ex_code_path)[1000:]}
                        """
        return autogen.AssistantAgent(
            name="Coder",
            system_message=system_message,
            llm_config=self.llm_config,
        )
    
    def _create_executor(self):
        # runs the code
        return autogen.UserProxyAgent(
            name="Executor",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "workspace",
                "use_docker": False,
            },
        )


    def _create_hypoth_brainstormer(self):
        system_message = f"""
                             You are an expert in galaxy formation. You must scan through recent literature in order to create hypotheses about galaxy formation. Each hypothesis must make
                             only one claim that will be tested. You may additionally include your reasoning behind that claim. These hypotheses will be investigated by creating plots from R
                             AMSES simulation data, so make sure your claims relate to that data. On your FIRST TURN ONLY, you should research information using your available search tools (listed below). 
                             Before tools that require a query, generate 5 interesting query ideas and select one at random. 

                             You have access to the following tool:
                             - search_arxiv(query: str): Search for papers on arXiv. Use simple queries like "galaxy formation" or "AGN feedback". You MUST use queries with 3 words or fewer.
                             
                             IMPORTANT: Even if the search results are not ideal, you MUST still generate hypotheses 
                             based on your knowledge of galaxy formation physics. Each hypothesis must make one testable claim.

                             If there is an additional promp given below, you must use it in order\
                             to create more relevant hypotheses. Please initally create 20 hypotheses. 

                             DO NOT generate new plot ideas - that's the PlotBrainstormer's job. 

                             Prompt: {self.prompt}
                             
                             If you are given feedback, please take it into account and alter your hypotheses appropriately. If this feedback advises you to entirely remove some of the 
                             hypotheses, remove them and DO NOT generate new ones to take their place. Our final goal is to narrow down our options until we have one great hypothesis. 

                             IMPORTANT: IF THIS IS NOT YOUR FIRST BRAINSTORM SESSION, please only utilize the information below. You may only search for new papers if you have a STRONG
                             suspicion that an idea could benefit from additional information.
                          """
        
        return ResearcherAgent(name="HypothesisBrainstormer", system_message=system_message, llm_config=self.llm_config, papers_per_query=2)
    
    def _create_critic(self):
        return autogen.AssistantAgent(
            name="Critic",
            system_message= f"""
                            You are an expert in galaxy formation. Please review the galaxy formation hypotheses and/or plot ideas given to you, and provide feedback on them. 
                            Our goal is to create one high quality hypothesis and one high quality plot that will help to prove or disprove that hypothesis. The plots will be made
                            using data from RAMSES simulations, so keep in mind the limitations of that data. Please also keep in mind that these plots should be possible to create
                            on a research quality laptop. We are dealing with large amounts of data, so some plots might be unrealistic to create in this enviroment. 

                            Each time you provide feedback, you must select 20%-40% of the ideas given to you as "advised for removal". Please consider each hypothesis 
                            and each plot to be one idea. (Ex: 10 hypotheses, each with 2 plot ideas = 10 hypotheses + 20 plot ideas = 30 ideas). If you remove a hypothesis, you
                            will simultaneously be removing all of the associated plots. Make sure that you will always leave at least one hypothesis and plot idea pair. 
                            When you get down to just a couple hypotheses and plot ideas, please additionally signify which one is your favorite. 

                            DO NOT generate new hypotheses - that's the HypothesisBrainstormer's job.
                            DO NOT generate new plot ideas - that's the PlotBrainstormer's job. 
                            DO NOT GENERATE CODE OR PSEUDO CODE. You must only provide hypothesis ideas in sentence/paragraph format.
                            MANDATORY WORKFLOW:
                                1. When you receive a request, search arxiv
                                2. IMMEDIATELY after receiving results: Generate 2-3 hypotheses
                                3. DO NOT STOP after the search - continue to hypothesis generation

                            The user made the following clarifications/requests:
                            {self.prompt}

                            IMPORTANT: When you get down to just a couple hypotheses and plot idea pairs (~3) and you are satisfied that we have ONE excellent 
                            hypothesis and ONE excellent plot idea, respond with your final recommendation (to provide to the coder) followed by the words 
                            "TERMINATE BRAINSTORM". 
                            """,
            llm_config=self.llm_config,
        )

    def _create_plot_brainstormer(self):
        system_message = f"""
                            You are a scientific researcher in galaxy formation. Your job is to take a list of hypotheses, and generate ideas for high quality 
                            plots that will help researchers evaluate the validity of those hypotheses. Your plot ideas must be given in the format of a paragraph. 
                            You should initially generate 3 plot ideas per hypothesis. Using the tool(s) listed below, you must search the literature about 
                            specific hypotheses and use this literature when creating plot ideas. Before using tools that require a query, generate 5 interesting
                            query ideas and select one at random. 

                            You have access to the following tool(s):
                            - search_arxiv(query: str): Search for papers on arXiv. Use simple queries like "galaxy formation" or "AGN feedback". You MUST use queries with 3 words or fewer.
                            
                            The user made the following clarifications/requests:
                            {self.prompt}

                            When given feedback on your ideas, you must take the feedback into account and improve your plot ideas. If this feedback advises you to 
                            remove some of your plot ideas, remove them and DO NOT generate new ones to take their place. Our final goal is to have one high quality hypothesis 
                            paired with one high quality plot idea. 

                            DO NOT GENERATE CODE OR PSEUDO CODE. You must only provide plot ideas in paragraph format.

                            Below are examples of code using MEGATRON data (a specific run of RAMSES). PLease use this to better understand what is capable with this data. 
                            Please also use your knowledge of the open source software package yt, as this will be available to the coder when creating the plots. 

                            {self._get_all_files_str(self.ex_code_path)}
                        """
        
        return ResearcherAgent(name="PlotBrainstormer", system_message=system_message, llm_config=self.llm_config)
    
    def _get_all_files_str(self, path):
        all_text = []
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and filename.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_text.append(f.read())

        # Combine into one string
        combined_text = "\n\n".join(all_text)
        return combined_text
