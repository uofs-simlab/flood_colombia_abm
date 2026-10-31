"""
The StudyArea class represents a geographical space for simulating flood events and their impacts. 
It extends Mesa-Geo's GeoSpace to integrate geospatial data with agent-based modeling.

Key features:
- Loads spatial data and converts it into agents (e.g., houses, businesses, schools, etc.).
- Manages flood areas and calculates flood impacts.
- Supports agent movement and spatial queries.

This class is designed for flexible and reusable flood simulation models with geospatial interactions.
"""

import uuid
import agents.flood_agents as FA
import random
import mesa_geo as mg
import geopandas as gpd
from shapely.geometry import Point, Polygon

class StudyArea(mg.GeoSpace):
    def __init__(self, model, houses_file, businesses_file, schools_file, shelter_file, healthcare_file, government_file, crs) -> None:
        super().__init__(crs=crs)

        self.model = model  # needed by get_flood_height_at_position (policy lever A)

        # Initialize attributes
        attributes = ["houses", "businesses", "schools", "healthcare", "shelter", "government", "flood_areas"]
        for attr in attributes:
            setattr(self, attr, [])
        
        # NOTE (Santa Lucia adaptation): the original Calgary version hardcoded
        # "EPSG:4269" here and force-overwrote every input layer's CRS with it
        # before reprojecting to `crs`. That was specific to how the Calgary
        # QGIS export lost its embedded CRS. The Santa Lucia shapefiles already
        # carry a correct, verified CRS (EPSG:32618 / UTM 18N) in their .prj
        # files, so forcing EPSG:4269 here would silently corrupt every
        # geometry (wrong-CRS reprojection). Instead we trust the CRS already
        # declared on disk and only reproject if `crs` differs from it.
        self.data_crs = None  # None => use each file's own declared CRS
        
        # Load different agent types using the generic function
        self._load_entity_agents_from_file(model, houses_file, FA.House_Agent, "houses", crs)
        self._load_entity_agents_from_file(model, businesses_file, FA.Business_Agent, "businesses", crs)
        self._load_entity_agents_from_file(model, schools_file, FA.School_Agent, "schools", crs)
        self._load_entity_agents_from_file(model, shelter_file, FA.Shelter_Agent, "shelter", crs)
        self._load_entity_agents_from_file(model, healthcare_file, FA.Healthcare_Agent, "healthcare", crs)
        self._load_entity_agents_from_file(model, government_file, FA.Government_Agent, "government", crs)

    

    def _load_entity_agents_from_file(self, model, file_path, agent_class, attr_name, crs) -> None:
        """
        Generic function to load entity agents from a file and assign UUIDs to each agent.
        
        Parameters:
        - model: The model instance to which agents belong.
        - file_path: The path to the file containing geographic data.
        - agent_class: The class of the agent to be created.
        - attr_name: The name of the attribute to store the agents in.
        - crs: The coordinate reference system to be used.
        """
        # Load and process data from the file
        df = gpd.read_file(file_path)
        if self.data_crs is not None:
            # Only used for source layers known to have a missing/incorrect
            # embedded CRS (not the case for the Santa Lucia data).
            df = df.set_crs(self.data_crs, allow_override=True)
        elif df.crs is None:
            raise ValueError(
                f"{file_path} has no CRS declared in its .prj and StudyArea.data_crs "
                "is None, so it cannot be safely reprojected. Set data_crs explicitly "
                "if this layer's CRS is really missing."
            )
        df = df.to_crs(crs)
        df = df[df.is_valid]  # Remove invalid geometries
        df["centroid"] = list(zip(df.geometry.centroid.x, df.geometry.centroid.y))
        
        # Generate UUIDs for agents
        df['unique_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        # Create agents from the GeoDataFrame with unique IDs
        agent_creator = mg.AgentCreator(agent_class, model=model, crs=crs)
        agents = agent_creator.from_GeoDataFrame(df, unique_id="unique_id")
        
        # Extend the relevant attribute with new agents
        getattr(self, attr_name).extend(agents)
        self.add_agents(agents)
        
        
    def _load_flood_maps_from_file(self, model, flood_file, crs) -> None:
        # Load the GeoDataFrame with the appropriate CRS
        flood_df = gpd.read_file(flood_file)
        if self.data_crs is not None:
            flood_df = flood_df.set_crs(self.data_crs, allow_override=True)
        elif flood_df.crs is None:
            raise ValueError(
                f"{flood_file} has no CRS declared in its .prj and StudyArea.data_crs "
                "is None, so it cannot be safely reprojected."
            )
        flood_df = flood_df.to_crs(crs)

        # Remove invalid, empty, or null geometries
        flood_df = flood_df[flood_df.is_valid]
        
        # Merge all features into one single polygon
        merged_polygon = flood_df.unary_union
        flood_df = gpd.GeoDataFrame(geometry=[merged_polygon], crs=crs)
    
        # Assign unique IDs to each feature
        # flood_df['id'] = range(1, len(flood_df) + 1)
    
        # Create FloodArea agents from the GeoDataFrame
        flood_creator = mg.AgentCreator(FloodArea, model, crs)
        flood_area = flood_creator.from_GeoDataFrame(flood_df)
    
        # Assign flood_file to each agent after creation
        for agent in flood_area:
            agent.flood_file = flood_file
    
        # Add the agents to the space and extend the flood_areas list
        self.add_agents(flood_area)
        self.flood_areas.extend(flood_area)
    
        # print(f"Number of flood areas added to the model: {len(flood_area)}")



            
    def remove_flood_maps(self, flood_file: str) -> None:
        # Find agents that match the flood_file
        agents_to_remove = [agent for agent in self.flood_areas if agent.flood_file == flood_file]
        
        # Remove each matching agent
        for agent in agents_to_remove:
            self.remove_agent(agent)
            self.flood_areas.remove(agent)


    def get_flood_height_at_position(self, position):
        """
        Check if the given position is within any flood area and return a random flood height if it is.

        Parameters:
        position (shapely.geometry.Point): The position to check.

        Returns:
        float: The flood height at the position, or 0 if the position is not flooded.

        Policy lever A (community-organized flood defense): if
        `model.community_organization_level` (0-1) is set above 0, the flood
        height drawn here is attenuated. This represents collective,
        organized local flood-fighting -- reinforcing embankments, blocking
        culverts/drains, continuous vigilance -- of the kind documented for
        Suan during the real 2010 Canal del Dique flood (in contrast to
        Santa Lucia, whose improvised defenses failed; see El Heraldo,
        "Santa Lucia y Suan, los contrastes de la catastrofe"). No hydraulic
        model of Santa Lucia was available to fit an exact attenuation
        magnitude, so the 0.5 maximum-effectiveness coefficient below is an
        explicit, swept modeling assumption -- results are reported across a
        range of `community_organization_level` values rather than resting
        on one unverified number (see manuscript Methods/Limitations).
        """
        defense_level = getattr(self.model, "community_organization_level", 0.0)
        # channel toggle + coefficient override, added for the Exp. A channel
        # ablation / coefficient sweep (manuscript v3, reviewer M2): default
        # values (True, 0.5) reproduce the original, undifferentiated behavior
        # exactly, so this is a strict superset, not a behavior change.
        depth_channel_on = getattr(self.model, "org_depth_channel_on", True)
        max_attenuation = getattr(self.model, "org_depth_attenuation_coef", 0.5)
        for flood_area in self.flood_areas:
            if flood_area.geometry.contains(position):
                raw_height = random.uniform(10, 55)
                if depth_channel_on:
                    return raw_height * (1 - max_attenuation * defense_level)
                return raw_height
        return 0.0
    
    
    def move_agent(self, agent, new_position):
        """
        Move the agent to a new position.
        
        Parameters:
        agent: The agent to move.
        new_position: A geometry object or an object with a geometry attribute.
        """
        if isinstance(new_position, (Point, Polygon)):
            new_pos = new_position.centroid
        elif hasattr(new_position, 'geometry'):
            new_pos = new_position.geometry.centroid
        else:
            raise ValueError("new_position must be a Point, Polygon, or an object with a geometry attribute.")
        
        agent.geometry = Point(new_pos.x, new_pos.y)
        

class FloodArea(mg.GeoAgent):
    def __init__(self, unique_id, model, geometry, crs, flood_file=None):
        super().__init__(unique_id, model, geometry, crs)
        self.flood_file = flood_file  # Assign the flood_file attribute
        # Initialize other necessary attributes