import uuid
from . import flood_agents as FA
from . import colombia_calibration as CAL

from scipy.stats import beta
import random
import numpy as np


def create_person_agents(model):
    """
    Create person agents in the model with demographics, wealth classes, and other attributes.

    Steps:
    1. Define proportions of different wealth classes and ethnicities.
    2. Shuffle and assign wealth classes and ethnicity groups to agents.
    3. Initialize agents with attributes: age, wealth, education, gender, mobility, etc.
    4. Add agents to the simulation's space and schedule.
    5. Assign agents to houses, businesses, and schools.

    Parameters:
    - model: The simulation model containing spatial agents and parameters.

    Santa Lucia adaptation: wealth-class and ethnicity proportions below are
    no longer the Calgary/Canada figures from the original model. They come
    from `colombia_calibration.py`, where every number is cited to a DANE (or
    equivalent) source. See that module's docstring for the full mapping
    between the original class/ethnicity labels and their new real-world
    meaning (Lower_Class=Pobre, Middle_Class=Vulnerable,
    Upper_Middle_Class=Clase media, Upper_Class=Clase alta; Indigenous=
    Indigena, Immigrant=Afrocolombiano, Canadian=Ninguna pertenencia declarada).
    """
    model.num_upper_class_persons = int(CAL.CLASE_ALTA_SHARE * model.num_persons)
    model.num_upper_middle_class_persons = int(CAL.CLASE_MEDIA_SHARE * model.num_persons)
    model.num_middle_class_persons = int(CAL.VULNERABLE_SHARE * model.num_persons)
    model.num_lower_class_persons = model.num_persons - model.num_upper_class_persons - model.num_upper_middle_class_persons - model.num_middle_class_persons

    model.wealth_classes = (
        ["Upper_Class"] * model.num_upper_class_persons +
        ["Upper_Middle_Class"] * model.num_upper_middle_class_persons +
        ["Middle_Class"] * model.num_middle_class_persons +
        ["Lower_Class"] * model.num_lower_class_persons
    )
    random.shuffle(model.wealth_classes)

    model.num_indigenous_persons = int(CAL.PERC_INDIGENA * model.num_persons)
    model.num_immigrant_persons = int(CAL.PERC_AFROCOLOMBIANO * model.num_persons)
    model.num_canadian_persons = model.num_persons - model.num_indigenous_persons - model.num_immigrant_persons

    ethnicity_groups = (
        ["Indigenous"] * model.num_indigenous_persons +
        ["Immigrant"] * model.num_immigrant_persons +
        ["Canadian"] * model.num_canadian_persons
    )
    random.shuffle(ethnicity_groups)

    model.num_working_class_persons = 0
    model.num_age_0_14_persons = 0
    model.num_age_15_64_persons = 0
    model.num_age_65_100_persons = 0
    model.num_male_persons = 0
    model.num_female_persons = 0

    model.persons_by_wealth_class = {wealth_class: [] for wealth_class in model.wealth_classes}

    for i in range(model.num_persons):
        unique_id = uuid.uuid4().int
        person = FA.Person_Agent(unique_id, model, geometry=None, crs = model.crs)
        person.population_index = i
        
        assign_age(model, i, person)
        assign_wealth(model, i, person)
        assign_mobility(model, person)
        person.ethnicity = ethnicity_groups[i]
        assign_education(model, person)
        assign_gender(model, person)
        
        model.space.add_agents(person)
        model.schedule.add(person)

        assign_working_class(model, person)
        assign_SES_index(model, person)

        model.persons_by_wealth_class[person.wealth_class].append(person)

    assign_persons_to_houses(model)
    assign_persons_to_businesses(model)
    assign_persons_to_schools(model)    


def assign_working_class(model, agent):
    if 18 <= agent.age <= 64:
        agent.working_class = True
        model.num_working_class_persons += 1           


def assign_education(model, person):
    if person.age >= 18:          
        if random.uniform(0,1) <= model.perc_education_people:
            person.education = 1  # Most educated with full understanding of flood risks
        else:
            person.education = random.uniform(0.5,0.9)   # Partial education
    else:
        person.education = 0.4/18*person.age     # Education scales with age for individuals under 18


def assign_gender(model, person):
    # Santa Lucia adaptation: DANE 2026 municipal projection, Santa Lucia
    # (colombia_calibration.py): 49.6% male / 50.4% female.
    perc_male = CAL.PERC_MALE
    if random.uniform(0,1) <= perc_male:
        model.num_male_persons += 1            
        person.gender = "Male"
    else:
        model.num_female_persons += 1
        person.gender = "Female"
        
     
def assign_age(model, i, person):
    # Santa Lucia adaptation: age-group cutoffs from DANE 2026 municipal
    # population projections for Santa Lucia (colombia_calibration.py),
    # replacing the original Canada-wide age distribution.
    cut_0_14 = int(CAL.AGE_0_14_SHARE * model.num_persons)
    cut_15_64 = int((CAL.AGE_0_14_SHARE + CAL.AGE_15_64_SHARE) * model.num_persons)
    if i < cut_0_14:	# Assign age for [0-14] age group
        person.age = random.randint(0, 14)
        model.num_age_0_14_persons += 1
    elif cut_0_14 <= i < cut_15_64:	# [15-64] age group
        person.age = random.randint(15, 64)
        model.num_age_15_64_persons += 1
    else:  				                # Assign age for [65-100] age group
        person.age = random.randint(65, 100)
        model.num_age_65_100_persons += 1


def assign_wealth(model, i, person):
    person.wealth_class = model.wealth_classes[i]  # Assign economy class based on shuffled list

    # Santa Lucia adaptation: annual income ranges (COP) come from DANE
    # per-capita monthly income bands for Pobre/Vulnerable/Clase
    # media/Clase alta (colombia_calibration.py), replacing the original
    # Canadian-dollar wealth-class ranges. Kept as "two weeks of annual
    # income" (x10/365) to preserve the original paper's savings-buffer
    # assumption (Table 6: initial wealth = yearly income/365*14).
    if person.wealth_class == "Upper_Class":
        lo, hi = CAL.ANNUAL_INCOME_UPPER_CLASS
    elif person.wealth_class == "Upper_Middle_Class":
        lo, hi = CAL.ANNUAL_INCOME_UPPER_MIDDLE_CLASS
    elif person.wealth_class == "Middle_Class":
        lo, hi = CAL.ANNUAL_INCOME_MIDDLE_CLASS
    elif person.wealth_class == "Lower_Class":
        lo, hi = CAL.ANNUAL_INCOME_LOWER_CLASS
    person.income = random.uniform(lo, hi) / 365 * 10  # two weeks' worth of annual income

    model.persons_gdp += person.income
    model.total_gdp += model.persons_gdp


def assign_SES_index(model, agent):   # High value represents high vulnerability
    # Age vulnerability
    if 0 <= agent.age <= 14:
        age_vul = 1
    elif 15 <= agent.age <= 64:
        age_vul = 0.1
    else:
        age_vul = 0.8
        
    # Education vulnerability 
    edu_vul = (1-0.9*agent.education)
    
    # Gender vulnerability
    if agent.gender == "Male":
        gen_vul = 0.1
    else:
        gen_vul = 1

    # Ethnicity vulnerability
    if agent.ethnicity == "Canadian":
        eth_vul = 0.1
    elif agent.ethnicity == "Immigrant":
        eth_vul = 0.8
    else:
        eth_vul = 1
    
    # Wealth status vulnerability
    if agent.wealth_class == "Upper_Class":
        wth_vul = 0.01
    elif agent.wealth_class == "Upper_Middle_Class":
        wth_vul = 0.1
    elif agent.wealth_class == "Middle_Class":
        wth_vul = 0.8
    else: 
        wth_vul = 1

    agent.SES_1 = (age_vul + edu_vul + gen_vul + eth_vul + wth_vul )/5
    agent.SES_2 = (age_vul * edu_vul * gen_vul * eth_vul * wth_vul )**(1/5) 
    agent.vulnerability = (agent.SES_1 + agent.SES_2) / 2
    

def assign_mobility(model, person):
    # Define parameters for the beta function of age
    a_age, b_age = 2, 5
    
    # Define parameters for the logistic function of wealth
    k_wealth, x0_wealth = 0.1, 50
    
    # Normalize age between 0 and 1
    x_age = person.age / 100
    
    # Compute mobility based on age using the beta function
    age_mobility = beta.pdf(x_age, a_age, b_age)
    
    # Compute mobility based on wealth using the logistic function
    wealth_mobility = 1 / (1 + np.exp(-k_wealth * (person.income - x0_wealth)))
    
    # Combine mobility from age and wealth
    person.mobility = age_mobility + wealth_mobility


def assign_persons_to_houses(model):
    """Construct balanced synthetic households with one adult anchor each."""
    houses = list(model.space.houses)
    persons = [p for group in model.persons_by_wealth_class.values() for p in group]
    if not houses:
        raise ValueError("Household allocation requires at least one house.")
    if len({id(p) for p in persons}) != len(persons):
        raise ValueError("Household allocation received duplicate people.")
    if len(persons) < len(houses):
        raise ValueError("Household allocation requires at least one person per house.")
    adults = [p for p in persons if p.age >= 18]
    if len(adults) < len(houses):
        raise ValueError("Household allocation requires at least one adult per house.")
    if any(house.residents for house in houses):
        raise ValueError("Household allocation requires initially empty resident lists.")

    random.shuffle(houses)
    random.shuffle(adults)
    anchors = adults[:len(houses)]
    anchor_ids = {id(p) for p in anchors}
    remaining = [p for p in persons if id(p) not in anchor_ids]
    random.shuffle(remaining)
    base_size, larger_houses = divmod(len(persons), len(houses))
    for index, (house, anchor) in enumerate(zip(houses, anchors)):
        size = base_size + (index < larger_houses)
        residents = [anchor] + [remaining.pop() for _ in range(size - 1)]
        for resident in residents:
            house.residents.append(resident)
            resident.household = house
            resident.homeless = False
            model.space.move_agent(resident, house.geometry)


def assign_persons_to_businesses(model):
    # Ensure there are businesses available
    if not model.space.businesses:
        print("No businesses available for assignment.")
        return
    
    for economy, persons in model.persons_by_wealth_class.items():
        for person in persons:
            # Assign only if the person is working class and not already employed
            if person.working_class and not person.employed:
                business = random.choice(model.space.businesses)
                business.employees.append(person)
                person.employed = True
                person.workplace = business


def assign_persons_to_schools(model):
    # Ensure there are schools available
    if not model.space.schools:
        print("No schools available for assignment.")
        return
    
    for economy, persons in model.persons_by_wealth_class.items():
        for person in persons:
            # Check if the person is within the school age range and not already assigned to a school
            if 5 <= person.age < 18 and not person.schoolplace:
                school = random.choice(model.space.schools)
                school.students.append(person)
                person.student = True
                person.schoolplace = school