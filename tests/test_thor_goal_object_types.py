import unittest
from types import SimpleNamespace

from alfworld.env.tasks import PickAndPlaceSimpleTask, PickTwoObjAndPlaceTask


def make_task(task_class, object_type, parent_type, sliced=False):
    # Goal checks only need a trajectory. The full initializer loads a scene graph.
    task = task_class.__new__(task_class)
    task.traj = {
        "pddl_params": {
            "object_target": object_type,
            "parent_target": parent_type,
            "toggle_target": None,
            "mrecep_target": None,
            "object_sliced": sliced,
        }
    }
    return task


def pickupable(object_type, object_id=None):
    return {
        "objectType": object_type,
        "objectId": object_id or f"{object_type}|1|2|3",
        "pickupable": True,
        "receptacle": False,
    }


def tray(object_type, *contents):
    return {
        "objectType": object_type,
        "objectId": f"{object_type}|4|5|6",
        "pickupable": False,
        "receptacle": True,
        "receptacleObjectIds": [obj["objectId"] for obj in contents],
    }


def scene(*objects):
    return SimpleNamespace(metadata={"objects": list(objects)})


class ThorGoalObjectTypesTest(unittest.TestCase):
    def test_substring_collision_does_not_satisfy_pick_and_place(self):
        for target, decoy in [
            ("Pen", "Pencil"),
            ("Knife", "ButterKnife"),
            ("Pot", "Potato"),
        ]:
            with self.subTest(target=target, decoy=decoy):
                task = make_task(PickAndPlaceSimpleTask, target, "Desk")
                similar = pickupable(decoy)
                self.assertFalse(
                    task.goal_satisfied(scene(tray("Desk", similar), similar))
                )

                exact = pickupable(target)
                self.assertTrue(task.goal_satisfied(scene(tray("Desk", exact), exact)))

    def test_sliced_object_type_is_distinct_from_parent_id(self):
        sliced = pickupable("AppleSliced", "Apple|1|2|3|AppleSliced_0")
        sliced_task = make_task(PickAndPlaceSimpleTask, "Apple", "Bowl", sliced=True)
        whole_task = make_task(PickAndPlaceSimpleTask, "Apple", "Bowl")
        self.assertTrue(sliced_task.goal_satisfied(scene(tray("Bowl", sliced), sliced)))
        self.assertFalse(whole_task.goal_satisfied(scene(tray("Bowl", sliced), sliced)))

        whole = pickupable("Apple")
        self.assertFalse(sliced_task.goal_satisfied(scene(tray("Bowl", whole), whole)))
        self.assertTrue(whole_task.goal_satisfied(scene(tray("Bowl", whole), whole)))

    def test_pick_two_without_target_receptacle_is_unsatisfied(self):
        first = pickupable("Pen", "Pen|1|2|3")
        second = pickupable("Pen", "Pen|2|3|4")
        task = make_task(PickTwoObjAndPlaceTask, "Pen", "Desk")
        self.assertEqual(
            task.goal_conditions_met(
                scene(tray("Shelf", first, second), first, second)
            ),
            (0, 2),
        )
        self.assertTrue(
            task.goal_satisfied(scene(tray("Desk", first, second), first, second))
        )

    def test_goal_checks_instance_ids_for_receptacle_membership(self):
        inside = pickupable("Pencil")
        outside = pickupable("Pen")
        task = make_task(PickAndPlaceSimpleTask, "Pen", "Desk")
        self.assertFalse(
            task.goal_satisfied(scene(tray("Desk", inside), inside, outside))
        )
        self.assertTrue(
            task.goal_satisfied(scene(tray("Desk", outside), inside, outside))
        )
