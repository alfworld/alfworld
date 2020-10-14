import re
import hashlib
import argparse
from termcolor import colored

import textworld
from textworld.logic import Proposition, Variable
from textworld.logic.logic2 import State, GameLogic

from textworld.generator.game2 import Game

import fast_downward



def load_task(domain_filename="domain.pddl", task_filename="problem.pddl"):
    # TODO: change translate.py so it's not relying on command line arguments.
    import sys

    from fast_downward.translate import pddl_parser
    # from fast_downward.translate import normalize
    # from fast_downward.translate import options
    # from fast_downward.translate import translate
    # sys.argv = ["translate.py", "domain", "task"]

    # options.domain = domain_filename
    # options.task = task_filename
    # options.filter_unimportant_vars = False
    # options.filter_unreachable_facts = False
    # options.use_partial_encoding = True

    # task = pddl_parser.open(domain_filename=options.domain, task_filename=options.task)
    task = pddl_parser.open(domain_filename=domain_filename, task_filename=task_filename)
    # normalize.normalize(task)
    return task


def _demangle_alfred_name(text):
    text = text.replace("_bar_", "|")
    text = text.replace("_minus_", "-")
    text = text.replace("_dot_", ".")
    text = text.replace("_comma_", ",")

    splits = text.split("_", 1)
    if len(splits) == 1:
        return text

    name, rest = splits
    m = hashlib.md5()
    m.update(rest.encode("utf-8"))
    return "{}_{}".format(name, m.hexdigest()[:6])


def clean_alfred_facts(facts):
    def _clean_fact(fact: textworld.logic.Proposition):
        args = [Variable(_demangle_alfred_name(arg.name), arg.type) for arg in fact.arguments]
        return Proposition(fact.name, args)

    facts = [_clean_fact(fact) for fact in facts if not fact.name.startswith("new-axiom@")]
    return facts


def main():
    task = load_task(args.domain, args.problem)

    name2type = {o.name: o.type_name for o in task.objects}
    def _atom2proposition(atom):
        if isinstance(atom, fast_downward.translate.pddl.conditions.Atom):
            if atom.predicate == "=":
                return None

            return Proposition(atom.predicate, [Variable(arg, name2type[arg]) for arg in atom.args])

        elif isinstance(atom, fast_downward.translate.pddl.f_expression.Assign):
            if atom.fluent.symbol == "total-cost":
                return None

            #name = "{}_{}".format(atom.fluent.symbol, atom.expression.value)
            name = "{}".format(atom.expression.value)
            return Proposition(name, [Variable(arg, name2type[arg]) for arg in atom.fluent.args])

    facts = [_atom2proposition(atom) for atom in task.init]
    facts = list(filter(None, facts))


    def _convert_variable(variable):
        if variable.name == "agent1":
            return Variable("P")

        elif variable.name.split("_", 1)[0] in ["apple", "tomato", "potato", "bread", "lettuce", "egg"]:
            return Variable(variable.name, "f")

        elif variable.name.split("_", 1)[0] in ["garbagecan", "cabinet", "container", "fridge", "microwave", "sink"]:
            return Variable(variable.name, "c")

        elif variable.name.split("_", 1)[0] in ["tabletop", "stoveburner"]:
            return Variable(variable.name, "s")

        elif variable.name.split("_", 1)[0] in ["bowl", "pot", "plate", "mug", "fork", "knife", "pan", "spoon"]:
            return Variable(variable.name, "o")

        elif variable.type == "location":
            return Variable(variable.name, "r")

        elif variable.type == "receptacle":
            return Variable(variable.name, "c")

        elif variable.type in ["otype", "rtype"]:
            return variable

        print("Unknown variable:", variable)
        return variable


    I = Variable("I")
    P = Variable("P")

    def _exists(name, *arguments):
        for f in facts:
            if f.name != name:
                continue

            if all(v1 is None or v1 == v2 for v1, v2 in zip(arguments, f.arguments)):
                return True

        return False


    def _convert_proposition(proposition):
        proposition = Proposition(proposition.name, [_convert_variable(a) for a in proposition.arguments])

        if proposition.name == "atlocation":
            return Proposition("at", (P, proposition.arguments[-1]))

        elif proposition.name == "receptacleatlocation":
            return Proposition("at", proposition.arguments)

        elif proposition.name == "objectatlocation":
            if _exists("inreceptacle", proposition.arguments[0], None):
                return Proposition("at", proposition.arguments)
            else:
                return None

        elif proposition.name == "inreceptacle":
            if proposition.arguments[-1].type == "s":
                return Proposition("on", proposition.arguments)

            return Proposition("in", proposition.arguments)

        elif proposition.name == "opened":
            return Proposition("open", proposition.arguments)

        elif proposition.name == "not_opened":
            return Proposition("closed", proposition.arguments)

        elif proposition.name == "holds":
            return Proposition("in", (proposition.arguments[0], I))

        elif proposition.name in ["openable", "checked", "full"]:
            return None  # TODO: support those attributes/states.

        elif proposition.name in ["objecttype", "receptacletype"]:
            return None

        elif str.isdigit(proposition.name):
            return Proposition("connected", proposition.arguments)


        print("Unknown fact:", proposition)
        return proposition

    facts = [_convert_proposition(f) for f in facts]
    facts = filter(None, facts)
    facts = clean_alfred_facts(facts)

    variables = {v.name: v for p in facts for v in p.arguments}

    # from textworld.generator.data import KnowledgeBase
    # textworld.render.visualize(State(KnowledgeBase.default().logic, facts), True)

    import glob
    logic = GameLogic()
    logic.load_domain(args.domain)
    for f in glob.glob("data/textworld_data/logic/*.twl2"):
        logic.import_twl2(f)

    state = State.from_pddl(logic, args.problem)
    game = Game(state, quests=[])
    for info in game.infos.values():
        info.name = _demangle_alfred_name(info.id)

    from pprint import pprint
    from textworld.envs.tw2 import TextWorldEnv
    from textworld.agents import HumanAgent

    infos = textworld.EnvInfos(admissible_commands=True)
    env = TextWorldEnv(infos)
    env.load(game)

    agent = HumanAgent(True)
    agent.reset(env)

    obs = env.reset()
    while True:
        #pprint(obs)
        print(obs.feedback)
        cmd = agent.act(obs, 0, False)
        if cmd == "STATE":
            print("\n".join(sorted(map(str, clean_alfred_facts(obs._facts)))))
            continue

        elif cmd == "ipdb":
            from ipdb import set_trace; set_trace()
            continue

        obs, _, _ = env.step(cmd)
        print(colored("\n".join(sorted(map(str, clean_alfred_facts(obs.effects)))), "yellow"))

    from ipdb import set_trace; set_trace()

    options = textworld.GameOptions()
    options.path = "tw_games/test.z8"
    options.force_recompile = True

    from ipdb import set_trace; set_trace()

    world = World.from_facts(facts, kb=options._kb)

    # Keep names and descriptions that were manually provided.
    used_names = set()
    for k, var_infos in game.infos.items():
        if k in variables:
            game.infos[k].name = variables[k].name

    # Use text grammar to generate name and description.
    import numpy as np
    from textworld.generator import Grammar
    grammar = Grammar(options.grammar, rng=np.random.RandomState(options.seeds["grammar"]))
    game.change_grammar(grammar)
    game.metadata["desc"] = "Generated with textworld.GameMaker."



    path = "/home/macote/src/TextWorld/textworld/generator/data/logic/look.twl2"
    with open(path) as f:
        document = f.read()

    actions, grammar = _parse_and_convert(document, rule_name="start2")

    # compile_game(game, options)
    game.grammar = grammar

    env = TextWorldEnv()
    env.load(game=game)
    state = env.reset()

    while True:
        print(state.feedback)
        cmd = input("> ")
        state, _, _ = env.step(cmd)

    from ipdb import set_trace; set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain")
    parser.add_argument("--problem")
    parser.add_argument("--output")
    args = parser.parse_args()

    main()