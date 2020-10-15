# Play


### TextWorld

Interact with TextWorld games:

```bash
$ cd $ALFRED_ROOT
$ python scripts/play_alfred_tw.py data/json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-14/trial_T20190908_231731_054988/ --domain data/alfred.pddl
```

Use Tab to auto complete from a list of candidate actions from the text-engine.

### THOR

Interactive with embodied games:

```bash
$ cd $ALFRED_ROOT
$ python scripts/play_alfred_thor.py data/json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-14/trial_T20190908_231731_054988/ --controller oracle_astar --debug
```

Use `--agent mrcnn_astar` with `--debug` to inspect the BUTLER controller (MaskRCNN detector & A* Navigator).

![](../media/play_screenshot.png)