import gradio as gr
from multi_agent import run_fact_checker

def fact_check_interface(claim):
    result = run_fact_checker(claim)
    # Build a multi-line string
    output_str = (
        f"**Claim**: {claim}\n\n"
      #  f"**Actual Truth**: {result['actual_truth']}\n\n"
        f"**Retrieved Fact**: {result['retrieved_fact']}\n\n"
        f"**Bias**: {result['bias']}\n\n"
        f"**Verification**: {result['verification']}\n\n"
        f"**Final Verdict**: {result['final_verdict']}"
    )
    return output_str

iface = gr.Interface(
    fn=fact_check_interface,
    inputs="text",
    outputs="markdown",
    title="AI Fact Checker",
    description="Enter a claim. We'll show the actual truth, bias, verification, and final verdict.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()
