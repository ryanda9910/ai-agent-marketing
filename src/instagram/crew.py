from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from instagram.tools import SearchTools
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Instagram():
	"""Instagram crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def market_researcher (self) -> Agent:
		return Agent(
			config=self.agents_config['market_researcher'],
			tools=[SearchTools.search_internet, SearchTools.search_instagram, SearchTools.open_page],
			verbose=True,
		)
	
	@agent
	def content_strategist(self) -> Agent:
		return Agent(
			config=self.agents_config['content_strategist'],
			verbose=True,
		)
	
	@agent
	def visual_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['visual_creator'],
		)
	@agent 
	def copywriter(self) -> Agent:
		return Agent(
			config=self.agents_config['copywriter'],
		)

	@task
	def market_research(self) -> Task:
		return Task(
			config=self.tasks_config['market_research'],
			agent=self.market_researcher,
			output_file="market_research.md",
		)
	
	@task
	def content_strategy(self) -> Task:
		return Task(
			config=self.tasks_config['content_strategy'],
			agent=self.content_strategist,
		)
	
	@task
	def visual_content_creation(self) -> Task:
		return Task(
			config=self.tasks_config['visual_content_creation'],
			agent=self.visual_creator,
		)
	
	@task
	def copywriting(self) -> Task:
		return Task(
			config=self.tasks_config['copywriting'],
			agent=self.copywriter,
		)
	
	@task
	def report_final_content_stratgey(self) -> Task:
		return Task(
			config=self.tasks_config['report_final_content_stratgey'],
			agent=self.content_strategist(),
			output_file="final_content_strategy.md",
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Instagram crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
