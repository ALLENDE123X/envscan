import verifiers as vf

def reward(completion, answer, **kw):
    text = completion[-1]["content"]
    return float(eval(text) == answer)

def load_environment():
    return vf.SingleTurnEnv(dataset=None, rubric=vf.Rubric(funcs=[reward]))
