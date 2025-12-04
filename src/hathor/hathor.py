import autogen
from .agents import ResearcherAgent
import arxiv
import os

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
                        Please generate a high quality galaxy formation hypothesis that can be evaluated using RAMSES simulation data. \
                        Please also generate a high quality plot idea that will use RAMSES data to evaluate that hypothesis.
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
            
            # Check if critic said "TERMINATE BRAINSTORMING"
            if last_speaker == self.critic and last_message:
                if "TERMINATE BRAINSTORM" in last_message.get("content", ""):
                    return self.coder
        
            # Otherwise, use default transition logic
            return "auto"  # Returns None to use allowed_transitions

        self.groupchat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=15,
            speaker_selection_method=custom_speaker_selection,
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed"
        )
        self.hathor_llm_config = {
            "config_list": self.config_list,
            "temperature": 0.1,
        }

        self.hathor = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.hathor_llm_config)

    def _create_coder(self):
        system_message = f"""
                            You are a Python expert. Write clean, efficient code. Make sure to include ledgible comments and docstrings.
                            You are also an expert on the RAMSES simulations. 
                            
                            The user made the following clarifications/requests:
                            {self.prompt}

                            All data can be found at {self.data_path}
                            
                            When possible, please use tools from the open source package known as yt.
                            You can also model your code after the example codes found below. 

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
                             AMSES simulation data, so make sure your claims relate to that data. On your first turn, you should research information using your available search tools (listed below). 
                             Before tools that require a query, generate 5 interesting query ideas and select one at random. 

                             You have access to the following tool:
                             - search_arxiv(query: str): Search for papers on arXiv. Use simple queries like "galaxy formation" or "AGN feedback". You MUST use queries with 3 words or fewer.
                             
                             If there is an additional promp given below, you must use it in order\
                             to create more relevant hypotheses. Please initally create 20 hypotheses. 

                             Prompt: {self.prompt}
                             
                             If given feedback, please take it into account and alter your hypotheses appropriately. If this feedback advises you to entirely remove some of the 
                             hypotheses, remove them and DO NOT generate new ones to take their place. Our final goal is to narrow down our options until we have one great hypothesis. 

                             THIS IS NOT YOUR FIRST BRAINSTORM SESSION. Please only utilize the information below. You may only search for new papers if you have a STRONG
                             suspicion that an idea could benefit from additional information. . 
                          """
        
        return ResearcherAgent(name="HypothesisBrainstormer", system_message=system_message, llm_config=self.llm_config, papers_per_query=2)
    
    def _create_critic(self):
        return autogen.AssistantAgent(
            name="Critic",
            system_message= f"""
                            You are an expert in galaxy formation. Please review the galaxy formation hypotheses and/or plot ideas given to you, and provide feedback on them. 
                            Our goal is to create one high quality hypothesis and one high quality plot that will help to prove or disprove that hypothesis. The plots will be made
                            using data from RAMSES simulations, so keep in mind the limitations of that data. Please also keep in mind that these plots should be possible to create
                            on a research quality laptop. We are dealing with alrge amounts of data, so some plots might be unrealistic to create in this enviroment. 

                            Each time you provide feedback, you must select 20%-40% of the ideas given to you as "advised for removal". Please consider each hypothesis 
                            and each plot to be one idea. (Ex: 10 hypotheses, each with 2 plot ideas = 10 hypotheses + 20 plot ideas = 30 ideas). If you remove a hypothesis, you
                            will simultaneously be removing all of the associated plots. Make sure that you will always leave at least one hypothesis and plot idea pair. 
                            When you get down to just a couple hypotheses and plot ideas, please additionally signify which one is your favorite. 

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
            if os.path.isfile(file_path):  # Skip subdirectories
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_text.append(f.read())

        # Combine into one string
        combined_text = "\n\n".join(all_text)
        return combined_text
