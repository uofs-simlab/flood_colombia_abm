"""Household, rescue, protection and recovery invariants."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))

from agents import person_agent_assign as assignment
from agents.flood_agents import Healthcare_Agent, Person_Agent, Shelter_Agent
from model.flood_model import FloodModel
from run_experimentA_main_sweep import FIXED


def person(index=0, age=20):
    return SimpleNamespace(
        unique_id=index, population_index=index, age=age, education=1,
        gender="Male", ethnicity="Canadian", wealth_class="Upper_Class",
        alive=True, evacuated=False, injured=False, stranded=True,
        time_stranded=2, time_injured=0, time_in_shelter=0,
        shelter_admission_hour=None,
        recovery_rate=.5, survivability_duration=100,
        household=SimpleNamespace(geometry=box(0, 0, 1, 1), flooded=False),
        geometry=Point(.5, .5), income=100,
    )


def support_model(rate=2):
    model = FloodModel.__new__(FloodModel, seed=1)
    model._rescue_throughput_per_hour = rate
    model._rescue_fraction = 0
    model._rescues_this_hour = 0
    model._rescue_slots_this_hour = 0
    model.rescues_total = 0
    model.rescued_person_ids = set()
    model.rescue_stranding_durations = []
    model.rescue_records = []
    model.hours_before_rescue = 2
    model.hours_before_healthcare = 0
    model.recovery_eligibility = .9
    model.offsite_care = True
    model.last_evacuation_time = 168
    model.schedule = SimpleNamespace(agents=[], time=180)
    model.shelter = SimpleNamespace(
        sheltered_agents=[], capacity_limit=10000, geometry=box(10, 10, 11, 11),
        current_hour=180, wealth=10000,
    )
    model.healthcare = SimpleNamespace(
        hospitalized_agents=[], capacity_limit=10, geometry=box(20, 20, 21, 21),
    )
    model.space = SimpleNamespace(
        flood_areas=[], move_agent=lambda agent, position: setattr(agent, "geometry", position),
    )
    return model


class HouseholdTests(unittest.TestCase):
    def assign(self, ages, houses=2):
        residents = [person(i, age) for i, age in enumerate(ages)]
        homes = [SimpleNamespace(residents=[], geometry=box(i, 0, i + 1, 1))
                 for i in range(houses)]
        model = SimpleNamespace(
            persons_by_wealth_class={"one": residents[::2], "two": residents[1::2]},
            space=SimpleNamespace(houses=homes, move_agent=lambda *args: None),
        )
        assignment.assign_persons_to_houses(model)
        return residents, homes

    def test_population_is_assigned_once_and_balanced(self):
        residents, homes = self.assign([25, 30, 40, 50, 5, 6, 8])
        entries = [p for home in homes for p in home.residents]
        self.assertCountEqual([id(p) for p in entries], [id(p) for p in residents])
        self.assertEqual(sorted(len(home.residents) for home in homes), [3, 4])
        for home in homes:
            self.assertTrue(any(p.age >= 18 for p in home.residents))
            for p in home.residents:
                self.assertIs(p.household, home)
                self.assertFalse(p.homeless)

    def test_exactly_one_available_adult_per_house(self):
        _, homes = self.assign([18, 19, 1, 2, 3, 4])
        self.assertTrue(all(sum(p.age >= 18 for p in h.residents) == 1 for h in homes))

    def test_insufficient_adults_rejected(self):
        with self.assertRaisesRegex(ValueError, "adult"):
            self.assign([20, 1, 2, 3])

    def test_missing_houses_rejected(self):
        with self.assertRaisesRegex(ValueError, "house"):
            self.assign([20], houses=0)

    def test_more_houses_than_people_rejected(self):
        with self.assertRaises(ValueError):
            self.assign([20], houses=2)

    def test_duplicate_input_people_rejected(self):
        p = person()
        model = SimpleNamespace(
            persons_by_wealth_class={"one": [p], "two": [p]},
            space=SimpleNamespace(houses=[SimpleNamespace(residents=[], geometry=None)],
                                  move_agent=lambda *args: None),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assignment.assign_persons_to_houses(model)

    def test_age_boundaries(self):
        for age, expected in [(0, 1), (1, 1), (14, 1), (15, .1),
                              (20, .1), (64, .1), (65, .8), (100, .8)]:
            with self.subTest(age=age):
                p = person(age=age)
                assignment.assign_SES_index(None, p)
                component = 5 * p.SES_1 - .1 - .1 - .1 - .01
                self.assertAlmostEqual(component, expected)


class RescueTests(unittest.TestCase):
    def test_fractional_capacity_conserved(self):
        for rate in [0, .1, .4, 1.5, 3.092783505154639, 15]:
            with self.subTest(rate=rate):
                model = support_model(rate)
                index = 0
                for hour in range(100):
                    model._begin_rescue_hour()
                    for _ in range(math.ceil(rate) + 2):
                        model.notify_shelter(person(index))
                        index += 1
                    self.assertEqual(model.rescues_total,
                                     math.floor((hour + 1) * rate + 1e-10))
                    self.assertGreaterEqual(model._rescue_fraction, 0)
                    self.assertLess(model._rescue_fraction, 1)

    def test_idle_whole_slots_expire(self):
        model = support_model(1.5)
        for _ in range(11):
            model._begin_rescue_hour()
        for i in range(10):
            model.notify_shelter(person(i))
        self.assertEqual(model.rescues_total, 1)

    def test_ineligible_people_do_not_consume_slots(self):
        for state in [{"alive": False}, {"evacuated": True}, {"time_stranded": 1}]:
            with self.subTest(state=state):
                model = support_model()
                model._begin_rescue_hour()
                p = person()
                p.__dict__.update(state)
                model.notify_shelter(p)
                self.assertEqual(model.rescues_total, 0)

    def test_hospital_patient_cannot_be_rescued(self):
        model = support_model()
        model._begin_rescue_hour()
        p = person()
        model.healthcare.hospitalized_agents.append(p)
        model.notify_shelter(p)
        self.assertEqual(model.rescues_total, 0)
        self.assertEqual(model.shelter.sheltered_agents, [])

    def test_duplicate_admission_is_not_counted(self):
        model = support_model()
        model._begin_rescue_hour()
        p = person()
        model.notify_shelter(p)
        p.time_stranded = 2
        model.notify_shelter(p)
        self.assertEqual(model.rescues_total, 1)
        self.assertEqual(len(model.rescued_person_ids), 1)
        self.assertEqual(model.rescue_stranding_durations, [2])
        self.assertEqual(model.rescue_records[0]["person_index"], 0)

    def test_readmission_counts_events_and_distinct_people_separately(self):
        model = support_model()
        model._begin_rescue_hour()
        p = person()
        model.notify_shelter(p)
        model.shelter.sheltered_agents.remove(p)
        p.time_stranded = 2
        model.notify_shelter(p)
        self.assertEqual(model.rescues_total, 2)
        self.assertEqual(len(model.rescued_person_ids), 1)

    def test_full_shelter_does_not_consume_a_slot(self):
        model = support_model()
        model.shelter.capacity_limit = 0
        model._begin_rescue_hour()
        model.notify_shelter(person())
        self.assertEqual(model._rescues_this_hour, 0)

    def test_admission_resets_shelter_clock_and_stranding(self):
        model = support_model()
        model._begin_rescue_hour()
        p = person()
        p.time_in_shelter = 20
        model.notify_shelter(p)
        self.assertEqual(p.time_in_shelter, 0)
        self.assertEqual(p.time_stranded, 0)
        self.assertFalse(p.stranded)
        self.assertTrue(model.shelter.geometry.covers(p.geometry))


class ProtectionTests(unittest.TestCase):
    def shelter_person(self, model, index=0, injured=False):
        p = person(index)
        p.stranded = False
        p.injured = injured
        p.geometry = Point(10.5, 10.5)
        p.shelter_admission_hour = model.schedule.time
        model.shelter.sheltered_agents.append(p)
        return p

    def test_home_in_footprint_prevents_return_even_if_building_flag_is_false(self):
        model = support_model()
        p = self.shelter_person(model)
        p.time_in_shelter = 12
        p.shelter_admission_hour = model.schedule.time - 12
        model.space.flood_areas = [SimpleNamespace(geometry=box(-1, -1, 2, 2))]
        Shelter_Agent.step(self.bind_shelter(model))
        self.assertIn(p, model.shelter.sheltered_agents)
        self.assertTrue(model.shelter.geometry.covers(p.geometry))

    def bind_shelter(self, model):
        model.shelter.model = model
        return model.shelter

    def test_footprint_boundary_also_prevents_return(self):
        model = support_model()
        p = person()
        model.space.flood_areas = [SimpleNamespace(geometry=box(1, 0, 2, 1))]
        self.assertFalse(model.can_return_home(p))

    def test_footprint_clear_returns_all_eligible_residents_home(self):
        model = support_model()
        people = [self.shelter_person(model, i) for i in range(3)]
        for p in people:
            p.time_in_shelter = 12
            p.shelter_admission_hour = model.schedule.time - 12
        Shelter_Agent.step(self.bind_shelter(model))
        self.assertEqual(model.shelter.sheltered_agents, [])
        for p in people:
            self.assertTrue(p.household.geometry.covers(p.geometry))
            self.assertEqual(p.time_in_shelter, 0)

    def test_minimum_stay_is_respected(self):
        model = support_model()
        p = self.shelter_person(model)
        Shelter_Agent.step(self.bind_shelter(model))
        self.assertIn(p, model.shelter.sheltered_agents)

    def test_return_requires_twelve_elapsed_hours_not_twelve_activations(self):
        model = support_model()
        p = self.shelter_person(model)
        shelter = self.bind_shelter(model)
        admission_hour = model.schedule.time
        for elapsed in range(12):
            model.schedule.time = admission_hour + elapsed
            Shelter_Agent.step(shelter)
            self.assertIn(p, shelter.sheltered_agents)
        model.schedule.time = admission_hour + 12
        Shelter_Agent.step(shelter)
        self.assertNotIn(p, shelter.sheltered_agents)

    def test_injured_resident_waits_when_healthcare_full(self):
        model = support_model()
        p = self.shelter_person(model, injured=True)
        p.time_in_shelter = 20
        model.healthcare.capacity_limit = 0
        Shelter_Agent.step(self.bind_shelter(model))
        model._advance_injury(p)
        self.assertEqual(p.time_injured, 1)
        self.assertIn(p, model.shelter.sheltered_agents)

    def test_healthcare_transfers_every_eligible_resident_once(self):
        model = support_model()
        people = [self.shelter_person(model, i, injured=True) for i in range(3)]
        Shelter_Agent.step(self.bind_shelter(model))
        self.assertEqual(model.shelter.sheltered_agents, [])
        self.assertEqual(len(model.healthcare.hospitalized_agents), 3)
        for p in people:
            self.assertIn(p, model.healthcare.hospitalized_agents)
            self.assertTrue(model.healthcare.geometry.covers(p.geometry))
            self.assertFalse(p.stranded)

    def test_protected_person_does_not_make_another_evacuation_decision(self):
        model = support_model()
        model.disaster_period = "during_flood"
        p = self.shelter_person(model)
        p.model = model
        p.current_hour = 180
        with patch("agents.flood_agents.decision.step") as decision:
            Person_Agent.step(p)
        decision.assert_not_called()
        self.assertFalse(p.evacuated)
        self.assertEqual(p.current_hour, 181)

    def test_dead_or_evacuated_people_cannot_enter_healthcare(self):
        for state in [{"alive": False}, {"evacuated": True}]:
            model = support_model()
            p = person()
            p.injured = True
            p.__dict__.update(state)
            model.receive_healthcare(p)
            self.assertEqual(model.healthcare.hospitalized_agents, [])

    def test_recovered_patient_returns_home_when_safe(self):
        model = support_model()
        p = person()
        p.injured = True
        model.receive_healthcare(p)
        self.assertNotIn(p, model.shelter.sheltered_agents)
        Healthcare_Agent.continued_healthcare(self.bind_healthcare(model), p)
        self.assertFalse(p.injured)
        self.assertEqual(model.healthcare.hospitalized_agents, [])
        self.assertTrue(p.household.geometry.covers(p.geometry))

    def bind_healthcare(self, model):
        model.healthcare.model = model
        model.healthcare.continued_healthcare = lambda p: Healthcare_Agent.continued_healthcare(
            model.healthcare, p)
        return model.healthcare

    def test_recovered_patient_transfers_to_shelter_when_home_unsafe(self):
        model = support_model()
        model.space.flood_areas = [SimpleNamespace(geometry=box(-1, -1, 2, 2))]
        p = person()
        p.injured = True
        model.receive_healthcare(p)
        Healthcare_Agent.continued_healthcare(self.bind_healthcare(model), p)
        self.assertNotIn(p, model.healthcare.hospitalized_agents)
        self.assertIn(p, model.shelter.sheltered_agents)
        self.assertFalse(p.injured)
        self.assertEqual(model.rescues_total, 0)
        self.assertTrue(model.shelter.geometry.covers(p.geometry))

    def test_recovered_patient_remains_protected_if_no_safe_destination(self):
        model = support_model()
        model.shelter.capacity_limit = 0
        model.space.flood_areas = [SimpleNamespace(geometry=box(-1, -1, 2, 2))]
        p = person()
        p.injured = True
        model.receive_healthcare(p)
        healthcare = self.bind_healthcare(model)
        Healthcare_Agent.continued_healthcare(healthcare, p)
        self.assertIn(p, healthcare.hospitalized_agents)
        self.assertFalse(p.injured)
        self.assertTrue(model.healthcare.geometry.covers(p.geometry))
        model.space.flood_areas = []
        Healthcare_Agent.continued_healthcare(healthcare, p)
        self.assertNotIn(p, healthcare.hospitalized_agents)

    def test_patient_death_removes_hospital_membership(self):
        model = support_model()
        p = person()
        p.injured = True
        p.recovery_rate = .95
        p.time_injured = 99
        model.receive_healthcare(p)
        Healthcare_Agent.continued_healthcare(self.bind_healthcare(model), p)
        model._advance_injury(p)
        self.assertFalse(p.alive)
        self.assertNotIn(p, model.healthcare.hospitalized_agents)
        self.assertFalse(p.stranded)

    def test_patient_list_mutation_does_not_skip_recovery(self):
        model = support_model()
        healthcare = self.bind_healthcare(model)
        healthcare.current_hour = 180
        healthcare.wealth = 10000
        model.space.businesses = []
        for i in range(4):
            p = person(i)
            p.injured = True
            model.receive_healthcare(p)
        with patch("agents.flood_agents.random.choice", return_value=SimpleNamespace(
                flooded=False, wealth=10000)):
            Healthcare_Agent.step(healthcare)
        self.assertEqual(healthcare.hospitalized_agents, [])


class ClinicalTests(unittest.TestCase):
    def patient(self):
        model = support_model()
        p = person()
        p.model = model
        p.current_hour = 180
        p.injured = True
        p.recovery_rate = .95
        p.time_injured = 99
        model.shelter.model = model
        model.healthcare.model = model
        return model, p

    def test_nonrecovering_deadline_is_identical_in_every_location(self):
        for location in ["exposed", "shelter", "hospital", "evacuated", "post_flood"]:
            with self.subTest(location=location):
                model, p = self.patient()
                if location == "shelter":
                    model._place_in_shelter(p)
                elif location == "hospital":
                    model.receive_healthcare(p)
                elif location == "evacuated":
                    p.evacuated = True
                elif location == "post_flood":
                    p.stranded = False
                model._advance_injury(p)
                self.assertEqual(p.time_injured, 100)
                self.assertFalse(p.alive)
                self.assertFalse(p.injured)
                self.assertFalse(p.stranded)
                self.assertNotIn(p, model.shelter.sheltered_agents)
                self.assertNotIn(p, model.healthcare.hospitalized_agents)

    def test_survival_boundary_and_once_per_hour_progression(self):
        model, p = self.patient()
        p.time_injured = 98
        model._advance_injury(p)
        self.assertTrue(p.alive)
        self.assertEqual(p.time_injured, 99)
        model._advance_injury(p)
        self.assertFalse(p.alive)
        self.assertEqual(p.time_injured, 100)

    def test_facility_and_person_activations_do_not_advance_injury_clock(self):
        model, p = self.patient()
        model._place_in_shelter(p)
        model.healthcare.capacity_limit = 0
        Shelter_Agent.step(model.shelter)
        Person_Agent.step(p)
        self.assertEqual(p.time_injured, 99)
        model.healthcare.capacity_limit = 1
        Shelter_Agent.step(model.shelter)
        Healthcare_Agent.continued_healthcare(model.healthcare, p)
        self.assertEqual(p.time_injured, 99)
        model._advance_injury(p)
        self.assertFalse(p.alive)

    def test_transfer_preserves_elapsed_injury_time(self):
        model, p = self.patient()
        model._place_in_shelter(p)
        Shelter_Agent.step(model.shelter)
        self.assertEqual(p.time_injured, 99)
        self.assertIn(p, model.healthcare.hospitalized_agents)
        model._advance_injury(p)
        self.assertFalse(p.alive)

    def test_hospital_and_offsite_care_have_the_same_recovery_eligibility(self):
        for evacuated in [False, True]:
            for propensity in [.9, .900001]:
                with self.subTest(evacuated=evacuated, propensity=propensity):
                    model, p = self.patient()
                    p.recovery_rate = propensity
                    p.evacuated = evacuated
                    if not evacuated:
                        model.receive_healthcare(p)
                        Healthcare_Agent.continued_healthcare(model.healthcare, p)
                    model._advance_injury(p)
                    self.assertEqual(p.alive, propensity <= .9)
                    if p.alive:
                        self.assertFalse(p.injured)
                        self.assertEqual(p.time_injured, 0)
                    self.assertEqual(model.healthcare.hospitalized_agents, [])

    def test_offsite_care_does_not_use_local_capacity(self):
        model, p = self.patient()
        model.healthcare.capacity_limit = 0
        p.evacuated = True
        p.recovery_rate = .5
        model._advance_injury(p)
        self.assertTrue(p.alive)
        self.assertFalse(p.injured)
        self.assertEqual(model.healthcare.hospitalized_agents, [])

    def test_recovery_resets_the_clock_before_another_injury_episode(self):
        model, p = self.patient()
        p.recovery_rate = .5
        self.assertTrue(model.recover_injury(p))
        p.injured = True
        model._advance_injury(p)
        self.assertTrue(p.alive)
        self.assertEqual(p.time_injured, 1)

    def test_dead_and_uninjured_people_do_not_accrue_injury_time(self):
        for state in [{"alive": False}, {"injured": False}]:
            model, p = self.patient()
            p.__dict__.update(state)
            model._advance_injury(p)
            self.assertEqual(p.time_injured, 99)

    def test_continuous_stranding_retains_its_separate_death_threshold(self):
        model, p = self.patient()
        model.shelter.capacity_limit = 0
        p.injury_duration = 12
        p.time_stranded = p.survivability_duration
        Person_Agent.stranded_behavior(p)
        self.assertFalse(p.alive)
        self.assertFalse(p.stranded)


class IntegrationTests(unittest.TestCase):
    def make_model(self, seed=12345):
        return FloodModel(**FIXED, seed=seed)

    def test_injury_deadline_applies_with_and_without_a_hospital_bed(self):
        for available_beds in [0, 1]:
            with self.subTest(available_beds=available_beds):
                model = self.make_model()
                p = next(p for h in model.space.houses for p in h.residents)
                p.injured = True
                p.recovery_rate = .95
                p.time_injured = p.survivability_duration - 1
                model._place_in_shelter(p)
                model.schedule.time = 180
                model.shelter.current_hour = 180
                model.healthcare.current_hour = 180
                model.healthcare.capacity_limit = available_beds
                model.step()
                self.assertFalse(p.alive)
                self.assertNotIn(p, model.shelter.sheltered_agents)
                self.assertNotIn(p, model.healthcare.hospitalized_agents)

    def test_full_population_households_have_required_shape(self):
        model = self.make_model()
        houses = model.space.houses
        entries = [p for h in houses for p in h.residents]
        self.assertEqual(len(entries), 972)
        self.assertEqual(len({id(p) for p in entries}), 972)
        self.assertEqual(sum(len(h.residents) == 3 for h in houses), 172)
        self.assertEqual(sum(len(h.residents) == 4 for h in houses), 114)
        for home in houses:
            self.assertTrue(any(p.age >= 18 for p in home.residents))
            self.assertTrue(all(p.household is home for p in home.residents))
        self.assertEqual(sorted(p.population_index for p in entries), list(range(972)))

    def test_same_seed_reproduces_initialization_and_steps(self):
        frames = []
        memberships = []
        for _ in range(2):
            model = self.make_model()
            memberships.append([[p.population_index for p in h.residents]
                                for h in model.space.houses])
            for _ in range(4):
                model.step()
            frames.append(model.datacollector.get_model_vars_dataframe())
        self.assertEqual(memberships[0], memberships[1])
        pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)

    def test_different_seed_changes_initial_population(self):
        populations = []
        for seed in [1, 2]:
            model = self.make_model(seed)
            populations.append([[p.age for p in h.residents] for h in model.space.houses])
        self.assertNotEqual(populations[0], populations[1])

    def test_no_shared_server_output(self):
        path = ROOT / "data_collection" / "serverrun_results.csv"
        self.assertFalse(path.exists())
        self.make_model().step()
        self.assertFalse(path.exists())

    def test_invalid_seed_rejected(self):
        for seed in [-1, 2**32, True, 1.5]:
            with self.assertRaises(ValueError):
                self.make_model(seed)


if __name__ == "__main__":
    unittest.main()
