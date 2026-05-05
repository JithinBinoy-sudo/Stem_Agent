from discovery.llm_introspection import DomainProfile

class PromptRewriter:
    def rewrite(self, profile: DomainProfile) -> str:
        workflow_steps = "\n".join(
            f"   {i+1}. {step}" for i, step in enumerate(profile.workflow)
        )
        criteria_list = "\n".join(
            f"   - {c}" for c in profile.quality_criteria
        )
        return (
            f"You are a {profile.domain} specialist agent.\n\n"
            f"For every task you receive, follow this expert workflow:\n"
            f"{workflow_steps}\n\n"
            f"Quality bar — your output will be judged on:\n"
            f"{criteria_list}\n\n"
            f"Always cite your sources explicitly. Never state a fact without linking it to a source."
        )
