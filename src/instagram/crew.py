from typing import Dict, Any
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from instagram.tools.search import SearchTools

class InstagramCrew(CrewBase):
    """
    Instagram crew for managing content strategy and creation
    using OpenAI-powered search tools.
    """
    
    def __init__(
        self, 
        agents_config: str = 'config/agents.yaml',
        tasks_config: str = 'config/tasks.yaml',
        openai_api_key: str = None
    ):
        """
        Initialize the Instagram crew.
        
        Args:
            agents_config (str): Path to agents configuration file
            tasks_config (str): Path to tasks configuration file
            openai_api_key (str, optional): OpenAI API key
        """
        super().__init__()
        self.agents_config = agents_config
        self.tasks_config = tasks_config
        self.search_tools = SearchTools(api_key=openai_api_key)

    @agent
    def market_researcher(self) -> Agent:
        """Creates a market researcher agent with OpenAI search capabilities"""
        return Agent(
            config=self.agents_config['market_researcher'],
            tools=[
                self.search_tools.search_internet,
                self.search_tools.search_instagram,
                self.search_tools.open_page
            ],
            verbose=True
        )

    @agent
    def content_strategist(self) -> Agent:
        """Creates a content strategist agent"""
        return Agent(
            config=self.agents_config['content_strategist'],
            verbose=True
        )

    @agent
    def visual_creator(self) -> Agent:
        """Creates a visual content creator agent"""
        return Agent(
            config=self.agents_config['visual_creator']
        )

    @agent
    def copywriter(self) -> Agent:
        """Creates a copywriter agent"""
        return Agent(
            config=self.agents_config['copywriter']
        )

    @task
    def market_research(self) -> Task:
        """Creates a market research task"""
        return Task(
            config=self.tasks_config['market_research'],
            agent=self.market_researcher,
            output_file="market_research.md"
        )

    @task
    def content_strategy(self) -> Task:
        """Creates a content strategy task"""
        return Task(
            config=self.tasks_config['content_strategy'],
            agent=self.content_strategist
        )

    @task
    def visual_content_creation(self) -> Task:
        """Creates a visual content creation task"""
        return Task(
            config=self.tasks_config['visual_content_creation'],
            agent=self.visual_creator
        )

    @task
    def copywriting(self) -> Task:
        """Creates a copywriting task"""
        return Task(
            config=self.tasks_config['copywriting'],
            agent=self.copywriter
        )

    @task
    def report_final_content_strategy(self) -> Task:  # Fixed typo in method name
        """Creates a final content strategy report task"""
        return Task(
            config=self.tasks_config['report_final_content_strategy'],  # Fixed typo in config key
            agent=self.content_strategist(),
            output_file="final_content_strategy.md"
        )

    @crew
    def crew(self) -> Crew:
        """
        Creates the Instagram crew with all agents and tasks
        
        Returns:
            Crew: Configured CrewAI crew instance
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=2
        )