# Reflection

Working on this project was initially challenging and required significant effort, but it was also engaging and highly realistic. It helped me understand how many business processes can be optimized using agentic systems, allowing humans to focus on tasks that truly require creativity, communication, and management skills while automated agents handle structured and repetitive work.

One major strength of the implemented agentic workflow is its ability to rapidly generate a comprehensive project foundation. In a real-world setting, creating user stories, feature breakdowns, engineering tasks, and risk assessments usually takes weeks of meetings with multiple stakeholders. In this workflow, a well-defined product specification enabled the agents to produce these artifacts efficiently and with built-in quality checks. The separation between worker agents and evaluator agents functioned like a small internal review team, ensuring consistency and correctness across outputs.

A key limitation of the workflow is its high sensitivity to prompt wording and agent instructions. Small changes in persona definitions or knowledge text could significantly alter the results, requiring careful tuning and repeated iterations. Unlike human teams, the agents cannot ask for clarification, which makes ambiguity in instructions more risky and time-consuming to debug.

One specific improvement would be to extend the workflow so that outputs from the Risk Manager automatically influence downstream planning, such as dynamically adjusting task structure and generating a Gantt chart or RACI matrix based on identified risks. This would make the agentic system more adaptive and closer to real-world project management practices.
