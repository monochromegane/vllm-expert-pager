def register() -> None:
    # Imported inside the function because it can only be imported after vLLM
    # has loaded.
    from vllm_expert_pager import experts

    experts.register()
