
# Installation

## TextWorld
You need a custom branch of TextWorld for now.

    pip install https://github.com/MarcCote/TextWorld/archive/1.5.new_grammar_pddl.zip

## Fast-downward
    git clone https://github.com/MarcCote/fast-downward.git -b debug
    pip install fast-downward/

# Interact with an ALFRED environment using text
Within the root folder of this project, run

    python scripts/play_alfred.py --domain tests/alfred_PutTask_domain.pddl --problem tests/alfred_problem_0_0.pddl

> **Note:** Use `<tab>` to show autocompletion for text commands.

# Modifying the text grammar

The production rules used to generated the text feedback when interacting with an ALFRED environment can be found in
`textworld_data/logic/alfred.twl2`.

# Testing the interface
Within the root folder of this project, run

    python tests/test_interface.py --domain tests/domain.pddl --problem tests/problem.pddl
