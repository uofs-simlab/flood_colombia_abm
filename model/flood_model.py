"""
Flood Model Simulation

This model simulates a flood scenario where agents (persons, homes, businesses, shelter system, healthcare, and government) 
interact within a grid-based environment. Each agent has specific attributes and behaviors, such as risk perception, 
movement, distress reactions, and financial transactions. The simulation tracks the effects of flooding on agent behavior, 
economic activities, rescue operations, and shelter systems.

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import random
import numpy as np
from shapely.geometry import Point

from space.flood_space import StudyArea
from mesa import Model
from agents import person_agent_assign as psn_agnt
from agents import colombia_calibration as CAL
from agents.flood_agents import Person_Agent

from mesa.time import RandomActivation

from data_collection import data_collect

class FloodModel(Model):
    """
    The Model Class defines the environment and interaction rules for the flood simulation. 
    It manages the grid-based environment, schedules agent actions, tracks time progression, 
    and monitors the effects of flooding on agents, their behaviors, and economic activities.
    """
    def __init__(self, N_persons,
                 shelter_cap_limit, healthcare_cap_limit, shelter_funding, healthcare_funding, pre_flood_days, flood_days, post_flood_days,
                 houses_file, businesses_file, schools_file, shelter_file, healthcare_file, government_file,
                 flood_file_1, flood_file_2, flood_file_3, model_crs,
                 # --- Santa Lucia policy-lever parameters (all default to the
                 # original model's implicit baseline behaviour, so existing
                 # calls without these kwargs are unaffected) ---
                 community_organization_level=0.0,   # Lever A: 0-1
                 evac_subsidy_frac=0.0,               # Lever C: 0-1
                 evac_subsidy_target_class=None,      # Lever C: None = all classes
                 warning_lead_time_days=7,            # Lever C: original hardcoded value was 7
                 shelter_route_distance_km=0.0,       # Lever D: real routed distance to shelter
                 shelter_route_is_water_crossing=False,  # Lever D: crossing-only route (boat, no road)
                 shelter_site_name=None,              # Lever D: human-readable label only, unused in logic
                 org_depth_channel_on=True,           # Lever A ablation: depth-attenuation channel on/off
                 org_depth_attenuation_coef=0.5,      # Lever A sweep: depth-attenuation coefficient override
                 org_sct_channel_on=True,             # Lever A ablation: social-capital-boost channel on/off
                 org_sct_boost_coef=0.3,              # Lever A sweep: SCT-boost coefficient override
                 rescue_boats_available=None,         # Lever D sensitivity: fleet size override (None = calibration default)
                 boat_capacity_per_trip=None,         # Lever D sensitivity: per-trip capacity override (None = calibration default)
                 ablation_config=None,
                 seed=None,
                 evacuation_success_rule="inherited",
                 injury_onset_scale=1.0,
                 survival_deadline_scale=1.0,
                 recovery_eligibility=0.9,
                 offsite_care=True):
        super().__init__()
        if evacuation_success_rule not in ("inherited", "probit"):
            raise ValueError("evacuation_success_rule must be 'inherited' or 'probit'.")
        for name, value in (("injury_onset_scale", injury_onset_scale),
                            ("survival_deadline_scale", survival_deadline_scale)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        if (isinstance(recovery_eligibility, bool)
                or not isinstance(recovery_eligibility, (int, float))
                or not math.isfinite(recovery_eligibility)
                or not 0 <= recovery_eligibility <= 1):
            raise ValueError("recovery_eligibility must be in [0, 1].")
        if not isinstance(offsite_care, bool):
            raise ValueError("offsite_care must be a boolean.")
        self.evacuation_success_rule = evacuation_success_rule
        self.injury_onset_scale = injury_onset_scale
        self.survival_deadline_scale = survival_deadline_scale
        self.recovery_eligibility = recovery_eligibility
        self.offsite_care = offsite_care
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("seed must be an integer in [0, 2**32).")
            random.seed(seed)
            np.random.seed(seed)
            self.random.seed(seed)
        self.run_seed = seed
        self.shelter_site_name = shelter_site_name
        self.ablation_config = ablation_config
        self.org_depth_channel_on = org_depth_channel_on
        self.org_depth_attenuation_coef = org_depth_attenuation_coef
        self.org_sct_channel_on = org_sct_channel_on
        self.org_sct_boost_coef = org_sct_boost_coef

        self.crs = model_crs

        # Policy-lever state must exist before StudyArea/person creation,
        # since Person_Agent.__init__ and get_flood_height_at_position read
        # it via getattr(model, ..., default).
        self.community_organization_level = community_organization_level
        self.evac_subsidy_frac = evac_subsidy_frac
        self.evac_subsidy_target_class = evac_subsidy_target_class

        self.space = StudyArea(self, houses_file, businesses_file, schools_file, shelter_file, healthcare_file, government_file, model_crs)

        self.total_days = pre_flood_days + flood_days + post_flood_days
        self.pre_flood_days = pre_flood_days
        self.flood_days = flood_days
        self.flood_file_1 = flood_file_1
        self.flood_file_2 = flood_file_2
        self.flood_file_3 = flood_file_3

        # Calculate times for flood maps
        self.disaster_period = None

        # Lever C (early-warning lead time): the original model hardcoded a
        # fixed 7-day evacuation window before flood onset. Exposing it as
        # `warning_lead_time_days` is what makes "early warning investment"
        # an actual testable policy lever, since sweeping pre_flood_days
        # alone (with this hardcoded at 7) only changes the length of
        # pre-disaster "normal life" simulated, not how much warning time
        # agents actually get before the flood.
        self.warning_lead_time_days = warning_lead_time_days
        self.evacuation_time = (self.pre_flood_days - warning_lead_time_days) * 24
        self.last_evacuation_time = self.pre_flood_days * 24
        
        self.hours_before_rescue = 2
        self.hours_before_healthcare = 0

        # Lever D (shelter distance realism): the base model's rescue-to-
        # shelter mechanism never used the shelter's actual location for
        # anything except cosmetically placing an already-rescued agent on
        # the map (see notify_shelter() and colombia_calibration.py Section
        # 7 for the full rationale and sourcing). This computes a realistic,
        # distance-dependent RESCUE THROUGHPUT RATE (people/hour the town's
        # small rescue-boat fleet can extract and shelter), rather than a
        # simple per-agent delay, because the real bottleneck in a village-
        # scale flood response is logistics capacity (limited boats making
        # repeated round trips), not any single trip being slow.
        self.shelter_route_distance_km = shelter_route_distance_km
        if shelter_route_is_water_crossing:
            one_way_hours = shelter_route_distance_km / CAL.BOAT_SPEED_KMH
        else:
            flood_km = min(shelter_route_distance_km, CAL.FLOOD_EXIT_DISTANCE_KM)
            dry_km = max(0.0, shelter_route_distance_km - CAL.FLOOD_EXIT_DISTANCE_KM)
            one_way_hours = flood_km / CAL.BOAT_SPEED_KMH + dry_km / CAL.DRY_ROAD_SPEED_KMH
        round_trip_hours = 2.0 * one_way_hours
        # Floor of 0.2h (12 min) even at distance=0: loading/unloading time.
        self._rescue_round_trip_hours = max(round_trip_hours, 0.2)
        # Fleet-size sensitivity (manuscript v3, reviewer M5): override args
        # default to None, which reproduces the original calibration values
        # exactly -- this is a strict superset of prior behavior.
        boats = rescue_boats_available if rescue_boats_available is not None else CAL.RESCUE_BOATS_AVAILABLE
        cap_per_trip = boat_capacity_per_trip if boat_capacity_per_trip is not None else CAL.BOAT_CAPACITY_PER_TRIP
        self._rescue_throughput_per_hour = (
            boats * cap_per_trip
        ) / self._rescue_round_trip_hours
        if not math.isfinite(self._rescue_throughput_per_hour) or self._rescue_throughput_per_hour < 0:
            raise ValueError("Rescue throughput must be finite and nonnegative.")
        self._rescue_fraction = 0.0
        self._rescue_slots_this_hour = 0
        self._rescues_this_hour = 0
        self.rescues_total = 0
        self.rescued_person_ids = set()
        self.rescue_stranding_durations = []
        self.rescue_records = []
        
        flood_interval = (self.flood_days * 24) // 6
        self.flood_map_add_times = [
            self.last_evacuation_time + 1,
            self.last_evacuation_time + 1 + flood_interval,
            self.last_evacuation_time + 1 + 2 * flood_interval
        ]
        self.flood_map_remove_times = [
            self.flood_map_add_times[0] + 3 * flood_interval,
            self.flood_map_add_times[1] + 3 * flood_interval,
            self.flood_map_add_times[2] + 3 * flood_interval
        ]
        
        # Santa Lucia adaptation: DANE 2005 Atlantico department illiteracy
        # rate (15+), used as a proxy (see colombia_calibration.py).
        self.perc_education_people = CAL.PERC_EDUCATION_PEOPLE
        self.schedule = RandomActivation(self)

        self.num_persons = N_persons

        self.num_houses = len(self.space.houses)
        self.num_businesses = len(self.space.businesses)
        self.num_schools = len(self.space.schools)

        self.shelter_cap_limit = shelter_cap_limit/100 * self.num_persons
        self.healthcare_cap_limit = healthcare_cap_limit/100 * self.num_persons

        # Share total gdp among population, businesses and government.
        # Santa Lucia adaptation: the original per-capita CAD constants
        # (22000/1200/4200/4200/7000) are scaled to COP via the Colombia:Canada
        # GDP-per-capita ratio -- see colombia_calibration.py for the full,
        # cited derivation (CAD_TO_COP_SCALE).
        self.business_gdp = (self.num_persons * CAL.BUSINESS_GDP_PER_CAPITA_COP)/365 * 14
        self.school_gdp = (self.num_persons * CAL.SCHOOL_GDP_PER_CAPITA_COP)/365 * 14
        self.shelter_gdp = (self.num_persons * CAL.SHELTER_GDP_PER_CAPITA_COP)/365 * 14
        self.healthcare_gdp = (self.num_persons * CAL.HEALTHCARE_GDP_PER_CAPITA_COP)/365 * 14
        self.government_gdp = (self.num_persons * CAL.GOVERNMENT_GDP_PER_CAPITA_COP)/365 * 14
        self.persons_gdp = 0
        self.total_gdp = self.business_gdp + self.school_gdp  + self.shelter_gdp + self.healthcare_gdp + self.government_gdp      
        
    #------------Initialize government agent-----------------------------------
        self.government = self.space.government[0]
        self.government.wealth = self.government_gdp
        self.schedule.add(self.government)
            
    #------------Initialize shelter system agent------------------------------
        self.shelter = self.space.shelter[0]
        self.shelter.wealth = self.shelter_gdp 
        self.shelter.capacity_limit = self.shelter_cap_limit
        self.schedule.add(self.shelter)
                      
    #------------Initialize healthcare system agent------------------------------
        self.healthcare = self.space.healthcare[0]
        self.healthcare.wealth = self.healthcare_gdp 
        self.healthcare.capacity_limit = self.healthcare_cap_limit
        self.schedule.add(self.healthcare) 
    
    #---------Initialize businesses, schools and houses--------------------
        self._initialize_businesses()

        # Initialize schools
        self._initialize_schools()
        
        # Initialize houses
        self._initialize_houses()
        
    #-----------------Initialize person agents--------------------------------
        psn_agnt.create_person_agents(self)
        
        data_collect.data_collection(self)      
        
    #-------------------------------Step function------------------------------
    def step(self):
        self._begin_rescue_hour()

        # Baseline, Pre, During. Post flood periods
        if self.evacuation_time <= self.schedule.time <= self.last_evacuation_time:
            self.disaster_period = 'pre_flood_evac_period'
        if self.last_evacuation_time < self.schedule.time < (self.pre_flood_days + self.flood_days) * 24:
            self.disaster_period = 'during_flood'
        if self.schedule.time >= (self.pre_flood_days + self.flood_days) * 24:                
            self.disaster_period = 'post_flood'
        
        # Handle flood map addition
        if self.schedule.time == self.flood_map_add_times[0]:
            self.add_flood_maps(self.flood_file_1)
        elif self.schedule.time == self.flood_map_add_times[1]:
            self.add_flood_maps(self.flood_file_2)
        elif self.schedule.time == self.flood_map_add_times[2]:
            self.add_flood_maps(self.flood_file_3)
        
        # Handle flood map removal
        if self.schedule.time == self.flood_map_remove_times[0]:
            self.remove_flood_maps(self.flood_file_3)         
        elif self.schedule.time == self.flood_map_remove_times[1]:
            self.remove_flood_maps(self.flood_file_2)
        elif self.schedule.time == self.flood_map_remove_times[2]:
            self.remove_flood_maps(self.flood_file_1)
    
        self.datacollector.collect(self)
        self.schedule.step()
        # One clinical update follows all actions, regardless of activation order.
        for agent in self.schedule.agents:
            if isinstance(agent, Person_Agent):
                self._advance_injury(agent)

    def recover_injury(self, patient):
        if (not patient.alive or not patient.injured
                or patient.recovery_rate > self.recovery_eligibility):
            return False
        patient.injured = False
        patient.time_injured = 0
        return True

    def _advance_injury(self, patient):
        if not patient.alive or not patient.injured:
            return
        if patient.evacuated and self.offsite_care and self.recover_injury(patient):
            return
        patient.time_injured += 1
        if patient.time_injured >= patient.survivability_duration:
            self._mark_dead(patient)

    def _mark_dead(self, person):
        person.alive = False
        person.injured = False
        person.stranded = False
        person.duringflood_coping_action_implemented = False
        person.preflood_non_evacuation_measure_implemented = False
        person.postflood_adaptation_measures_planned = False
        for members in [self.shelter.sheltered_agents, self.healthcare.hospitalized_agents]:
            if person in members:
                members.remove(person)
        person.time_in_shelter = 0
        person.shelter_admission_hour = None
    
    def _begin_rescue_hour(self):
        credit = self._rescue_fraction + self._rescue_throughput_per_hour
        self._rescue_slots_this_hour = math.floor(credit + 1e-12)
        # Idle whole slots expire; only fractional capacity carries forward.
        self._rescue_fraction = max(0.0, credit - self._rescue_slots_this_hour)
        self._rescues_this_hour = 0
    
    def add_flood_maps(self, flood_file):
        self.space._load_flood_maps_from_file(self, flood_file, self.crs)

    def remove_flood_maps(self, flood_file):  # remove flood areas as were added
        self.space.remove_flood_maps(flood_file)

    #-------------------------Initialize businesses----------------------------
    
    def _initialize_businesses(self):
        avg_business_wealth = self.business_gdp / self.num_businesses    
        
        for business in self.space.businesses:
            business.wealth = avg_business_wealth
        
            self.schedule.add(business)

    #---------------------------Initialize schools-----------------------------
    
    def _initialize_schools(self):
        # School GDP share
        avg_school_wealth = self.school_gdp / self.num_schools
        
        # Assign all types to each school and distribute GDP
        for school in self.space.schools:
            school.wealth = avg_school_wealth
            
            # Add school agent to the schedule
            self.schedule.add(school)

    def _initialize_houses(self):
        # Assign random resilience to houses
        for house in self.space.houses:
            self.schedule.add(house)    
    
    #-------------Notify Shelter and get healthcare----------------------------
                       
    def notify_shelter(self, agent):
        if (not agent.alive or agent.evacuated
                or agent in self.shelter.sheltered_agents
                or agent in self.healthcare.hospitalized_agents
                or agent.time_stranded < self.hours_before_rescue
                or len(self.shelter.sheltered_agents) >= self.shelter.capacity_limit
                or self._rescues_this_hour >= self._rescue_slots_this_hour):
            return False
        duration = agent.time_stranded
        self._place_in_shelter(agent)
        self._rescues_this_hour += 1
        self.rescues_total += 1
        self.rescued_person_ids.add(agent.unique_id)
        self.rescue_stranding_durations.append(duration)
        self.rescue_records.append({
            "hour": self.schedule.time,
            "person_index": agent.population_index,
            "stranding_hours": duration,
        })
        return True

    def _place_in_shelter(self, agent):
        position = self.get_random_point_in_polygon(self.shelter.geometry)
        self.space.move_agent(agent, position)
        self.shelter.sheltered_agents.append(agent)
        agent.stranded = False
        agent.time_stranded = 0
        agent.time_in_shelter = 0
        agent.shelter_admission_hour = self.schedule.time

    def can_return_home(self, agent):
        return (agent.household is not None and not any(
            area.geometry.intersects(agent.household.geometry)
            for area in self.space.flood_areas))

    def return_home(self, agent):
        if not agent.alive or not self.can_return_home(agent):
            return False
        position = self.get_random_point_in_polygon(agent.household.geometry)
        self.space.move_agent(agent, position)
        if agent in self.shelter.sheltered_agents:
            self.shelter.sheltered_agents.remove(agent)
        if agent in self.healthcare.hospitalized_agents:
            self.healthcare.hospitalized_agents.remove(agent)
        agent.time_in_shelter = 0
        agent.shelter_admission_hour = None
        agent.stranded = False
        agent.time_stranded = 0
        return True

    def discharge_patient(self, patient):
        if patient not in self.healthcare.hospitalized_agents:
            raise ValueError("Cannot discharge a person who is not hospitalized.")
        if not patient.alive or patient.injured:
            return False
        if self.return_home(patient):
            return True
        if len(self.shelter.sheltered_agents) < self.shelter.capacity_limit:
            self._place_in_shelter(patient)
            self.healthcare.hospitalized_agents.remove(patient)
            return True
        # A recovered patient still occupies a bed until a safe destination exists.
        return False
    
    def get_random_point_in_polygon(self, polygon):
        """Generate a random point within a given polygon."""
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(random_point):
                return random_point
                 
    def receive_healthcare(self, patient):
        if (not patient.alive or patient.evacuated or not patient.injured
                or patient.time_injured < self.hours_before_healthcare
                or patient in self.healthcare.hospitalized_agents
                or len(self.healthcare.hospitalized_agents) >= self.healthcare.capacity_limit):
            return False
        position = self.get_random_point_in_polygon(self.healthcare.geometry)
        self.space.move_agent(patient, position)
        self.healthcare.hospitalized_agents.append(patient)
        if patient in self.shelter.sheltered_agents:
            self.shelter.sheltered_agents.remove(patient)
        patient.stranded = False
        patient.time_stranded = 0
        patient.time_in_shelter = 0
        patient.shelter_admission_hour = None
        return True
