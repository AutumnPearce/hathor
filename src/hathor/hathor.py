import autogen
from .agents import BrainstormerAgent


TEST_DATA_PATH = "../../test_data"
EX_CODE_PATH = "../../ex_code"

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
        user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
         
        user_proxy.initiate_chat(
            self.hathor,
            message="Please create 20 hypotheses about galaxy formation."
        )
        
        return self.groupchat.messages

    def respond(self, response):
        pass
        
    def _setup_groupchat(self):
        self.brainstormer = self._create_brainstormer()
        # self.plot_brainstormer = self._create_plot_brainstormer()
        self.critic = self._create_critic()
        # self.hpc_expert = self._create_hpc_expert()
        # self.coder = self._create_coder()
        # self.debugger = self._create_debugger()
        # self.executor = self._create_manager()

        self.agents = [self.brainstormer, self.critic]

        self.groupchat = groupchat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=15,
            speaker_selection_method="auto"
        )
        self.hathor = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

    def _create_brainstormer(self):
        system_message = f"""
                             You are an expert in galaxy formation. You must scan through recent literature in order to create hypotheses about galaxy formation
                             that can be investigated by creating plots of RAMSES simulation data. If there is an additional promp given below, you must use it in order\
                             to make more relevant hypotheses. Please create 20 hypotheses. 

                             Prompt: {self.prompt}
                          """
        return BrainstormerAgent(name="brainstormer", system_message=system_message, llm_config=self.llm_config)
    
    def _create_critic(self):
        return autogen.AssistantAgent(
            name="Critic",
            system_message="""You review python script solutions. Check for errors and suggest improvements. You have extensive python knowledge, so please keep an eye out for python errors. """,
            llm_config=self.llm_config,
        )

