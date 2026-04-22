"""
3D Visualization Module (Enhanced)
Creates interactive 3D visualizations using Plotly
Visualizes realistic building, tank, rain animation, and humanoid workers
"""

import math
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Tuple
import config


class Scene3D:
    """
    Creates and manages 3D scene visualization for the simulation.
    
    Components:
    - Realistic building with multiple levels, windows, and roof
    - Storage tank with dynamic fill level
    - Animated rain particles with intensity-based count
    - Humanoid worker agents (head, body, limbs)
    """
    
    def __init__(self):
        """Initialize the 3D scene."""
        self.fig = None
        self.camera_position = dict(
            x=30, y=30, z=25
        )
        
    def create_realistic_building(self) -> List[go.Mesh3d]:
        """
        Create a realistic building with multiple levels, windows, and roof.
        
        Returns:
            List of Plotly Mesh3d objects representing building components
        """
        building_parts = []
        
        # Main building structure (walls)
        x_walls = [0, config.BUILDING_WIDTH, config.BUILDING_WIDTH, 0,
                   0, config.BUILDING_WIDTH, config.BUILDING_WIDTH, 0]
        y_walls = [0, 0, config.BUILDING_DEPTH, config.BUILDING_DEPTH,
                   0, 0, config.BUILDING_DEPTH, config.BUILDING_DEPTH]
        z_walls = [0, 0, 0, 0,
                   config.BUILDING_HEIGHT, config.BUILDING_HEIGHT, config.BUILDING_HEIGHT, config.BUILDING_HEIGHT]
        
        i_walls = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j_walls = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k_walls = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        
        walls = go.Mesh3d(
            x=x_walls, y=y_walls, z=z_walls,
            i=i_walls, j=j_walls, k=k_walls,
            name='Building Walls',
            color='#A9A9A9',
            opacity=0.85,
            showlegend=True
        )
        building_parts.append(walls)
        
        # Roof - triangular pyramid on top
        roof_height = config.BUILDING_HEIGHT
        roof_x = [0, config.BUILDING_WIDTH, config.BUILDING_WIDTH, 0,
                  config.BUILDING_WIDTH / 2]
        roof_y = [0, 0, config.BUILDING_DEPTH, config.BUILDING_DEPTH,
                  config.BUILDING_DEPTH / 2]
        roof_z = [roof_height, roof_height, roof_height, roof_height,
                  roof_height + 3]  # Peak at 3m above walls
        
        roof_i = [0, 1, 2, 3, 0, 1, 2, 3]
        roof_j = [1, 2, 3, 0, 4, 4, 4, 4]
        roof_k = [4, 4, 4, 4, 1, 2, 3, 0]
        
        roof = go.Mesh3d(
            x=roof_x, y=roof_y, z=roof_z,
            i=roof_i, j=roof_j, k=roof_k,
            name='Roof',
            color='#DC143C',
            opacity=0.9,
            showlegend=True
        )
        building_parts.append(roof)
        
        # Add windows as simple rectangular frames
        windows = self._create_windows()
        building_parts.extend(windows)
        
        return building_parts
    
    def _create_windows(self) -> List[go.Scatter3d]:
        """
        Create window representations on building walls.
        
        Returns:
            List of Scatter3d objects representing windows
        """
        windows = []
        window_width = 1.5
        window_height = 1.2
        
        # Front wall windows (y=0)
        for level in range(3):  # 3 levels of windows
            z_pos = 3 + level * 3.5
            for col in range(3):  # 3 columns of windows
                x_pos = 2 + col * 5.5
                
                # Window frame
                x_frame = [x_pos - window_width/2, x_pos + window_width/2, 
                          x_pos + window_width/2, x_pos - window_width/2,
                          x_pos - window_width/2]
                y_frame = [0, 0, 0, 0, 0]
                z_frame = [z_pos - window_height/2, z_pos - window_height/2,
                          z_pos + window_height/2, z_pos + window_height/2,
                          z_pos - window_height/2]
                
                window = go.Scatter3d(
                    x=x_frame, y=y_frame, z=z_frame,
                    mode='lines',
                    line=dict(color='#87CEEB', width=2),
                    name='Windows',
                    showlegend=False
                )
                windows.append(window)
        
        return windows
    
    def create_realistic_tank(self, tank_level_percentage: float = 50) -> go.Surface:
        """
        Create a cylindrical storage tank with dynamic fill level and base.
        
        Args:
            tank_level_percentage: Fill level as percentage (0-100)
            
        Returns:
            Plotly surface object representing tank
        """
        # Position tank next to building
        tank_x_center = config.BUILDING_WIDTH + 8
        tank_y_center = config.BUILDING_DEPTH / 2
        
        # Create cylinder using parametric equations
        theta = np.linspace(0, 2*np.pi, 40)
        z_base = np.linspace(0, config.TANK_HEIGHT, 25)
        
        Theta, Z = np.meshgrid(theta, z_base)
        X = tank_x_center + config.TANK_RADIUS * np.cos(Theta)
        Y = tank_y_center + config.TANK_RADIUS * np.sin(Theta)
        
        # Water fill level calculation
        fill_height = config.TANK_HEIGHT * tank_level_percentage / 100
        
        # Create colormap: water is blue, empty part is gray
        colors = np.ones_like(Z)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z[i, j] <= fill_height:
                    colors[i, j] = 0  # Water (blue)
                else:
                    colors[i, j] = 1  # Empty (gray)
        
        tank = go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=colors,
            colorscale=[[0, '#1E90FF'], [1, '#D3D3D3']],
            name='Tank',
            showlegend=True,
            showscale=False,
            hovertemplate='Tank<extra></extra>'
        )
        
        return tank
    
    def create_tank_base(self) -> go.Mesh3d:
        """
        Create a circular base plate for the tank.
        
        Returns:
            Plotly Mesh3d object for tank base
        """
        tank_x_center = config.BUILDING_WIDTH + 8
        tank_y_center = config.BUILDING_DEPTH / 2
        
        # Create circular base
        theta = np.linspace(0, 2*np.pi, 20)
        x_base = tank_x_center + (config.TANK_RADIUS + 0.2) * np.cos(theta)
        y_base = tank_y_center + (config.TANK_RADIUS + 0.2) * np.sin(theta)
        z_base = np.zeros_like(theta)
        
        # Add center point
        x_base = list(x_base) + [tank_x_center]
        y_base = list(y_base) + [tank_y_center]
        z_base = list(z_base) + [0]
        
        base = go.Scatter3d(
            x=x_base, y=y_base, z=z_base,
            mode='markers',
            marker=dict(size=1, color='#505050'),
            name='Tank Base',
            showlegend=False
        )
        
        return base
    
    def create_animated_rain_particles(
        self,
        rain_intensity: float,
        frame_index: int = 0,
        num_particles: int = None
    ) -> go.Scatter3d:
        """
        Create animated falling rain particles with intensity-based count.
        
        Rain falls from top to bottom, updating position based on frame.
        
        Args:
            rain_intensity: Rain intensity (0-100 mm)
            frame_index: Animation frame number (for falling effect)
            num_particles: Number of particles to render
            
        Returns:
            Plotly Scatter3d object for rain
        """
        if num_particles is None:
            # Scale particle count with rain intensity
            num_particles = int((rain_intensity / 50) * config.RAIN_PARTICLE_COUNT_MAX)
            num_particles = max(0, min(num_particles, config.RAIN_PARTICLE_COUNT_MAX))
        
        if rain_intensity < 0.1 or num_particles == 0:
            # No rain - return empty scatter
            return go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(size=2, color=config.COLOR_RAIN),
                name='Rain',
                showlegend=True,
                hovertemplate='<extra></extra>'
            )
        
        # Create random rain particles with animation effect
        np.random.seed(42)
        base_x = np.random.uniform(-2, config.BUILDING_WIDTH + 5, num_particles)
        base_y = np.random.uniform(-2, config.BUILDING_DEPTH + 2, num_particles)
        
        # Animate rain falling - particles cycle through heights
        fall_speed = 15  # Units per frame
        max_height = config.BUILDING_HEIGHT + 10
        
        # Calculate z position with falling animation
        z = np.zeros_like(base_x)
        for i in range(num_particles):
            # Each particle has different start height
            particle_phase = (frame_index + i) % 100
            z[i] = max_height - (particle_phase * fall_speed / 100) % max_height
        
        # Rain color intensity based on rainfall
        opacity = min(0.8, 0.3 + (rain_intensity / 100) * 0.5)
        
        rain = go.Scatter3d(
            x=base_x, y=base_y, z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=config.COLOR_RAIN,
                opacity=opacity,
                line=dict(width=0)
            ),
            name='Rain',
            showlegend=True,
            hovertemplate='Rain droplet<extra></extra>'
        )
        
        return rain
    
    def create_humanoid_worker(
        self,
        position: Tuple[float, float, float],
        worker_id: int = 0
    ) -> List[go.Scatter3d]:
        """
        Create a humanoid worker model with head, body, and limbs.
        
        Args:
            position: (x, y, z) center position of worker
            worker_id: Worker ID for identification
            
        Returns:
            List of Scatter3d objects representing worker body parts
        """
        x_pos, y_pos, z_pos = position
        worker_parts = []
        
        # Head (sphere-like using scatter)
        head_radius = 0.3
        head_z = z_pos + 1.5
        head = go.Scatter3d(
            x=[x_pos], y=[y_pos], z=[head_z],
            mode='markers',
            marker=dict(
                size=12,
                color='#FFD7A8',  # Skin color
                symbol='circle'
            ),
            name=f'Worker {worker_id}',
            showlegend=False,
            hovertemplate=f'Worker {worker_id}<extra></extra>'
        )
        worker_parts.append(head)
        
        # Body (vertical line from head to waist)
        body_top_z = z_pos + 1.2
        body_bottom_z = z_pos + 0.3
        body = go.Scatter3d(
            x=[x_pos, x_pos],
            y=[y_pos, y_pos],
            z=[body_top_z, body_bottom_z],
            mode='lines',
            line=dict(color='#FF4500', width=6),  # Orange shirt
            name=f'Worker {worker_id} Body',
            showlegend=False,
            hovertemplate='<extra></extra>'
        )
        worker_parts.append(body)
        
        # Left arm
        left_arm_x = [x_pos, x_pos - 0.35]
        left_arm_y = [y_pos, y_pos]
        left_arm_z = [z_pos + 0.9, z_pos + 0.7]
        left_arm = go.Scatter3d(
            x=left_arm_x, y=left_arm_y, z=left_arm_z,
            mode='lines+markers',
            line=dict(color='#FFD7A8', width=4),
            marker=dict(size=5, color='#FFD7A8'),
            showlegend=False,
            hovertemplate='<extra></extra>'
        )
        worker_parts.append(left_arm)
        
        # Right arm
        right_arm_x = [x_pos, x_pos + 0.35]
        right_arm_y = [y_pos, y_pos]
        right_arm_z = [z_pos + 0.9, z_pos + 0.7]
        right_arm = go.Scatter3d(
            x=right_arm_x, y=right_arm_y, z=right_arm_z,
            mode='lines+markers',
            line=dict(color='#FFD7A8', width=4),
            marker=dict(size=5, color='#FFD7A8'),
            showlegend=False,
            hovertemplate='<extra></extra>'
        )
        worker_parts.append(right_arm)
        
        # Left leg
        left_leg_x = [x_pos - 0.15, x_pos - 0.15]
        left_leg_y = [y_pos, y_pos]
        left_leg_z = [z_pos + 0.3, z_pos - 0.3]
        left_leg = go.Scatter3d(
            x=left_leg_x, y=left_leg_y, z=left_leg_z,
            mode='lines+markers',
            line=dict(color='#2F4F4F', width=4),  # Dark pants
            marker=dict(size=5, color='#FFD7A8'),
            showlegend=False,
            hovertemplate='<extra></extra>'
        )
        worker_parts.append(left_leg)
        
        # Right leg
        right_leg_x = [x_pos + 0.15, x_pos + 0.15]
        right_leg_y = [y_pos, y_pos]
        right_leg_z = [z_pos + 0.3, z_pos - 0.3]
        right_leg = go.Scatter3d(
            x=right_leg_x, y=right_leg_y, z=right_leg_z,
            mode='lines+markers',
            line=dict(color='#2F4F4F', width=4),
            marker=dict(size=5, color='#000000'),
            showlegend=False,
            hovertemplate='<extra></extra>'
        )
        worker_parts.append(right_leg)
        
        return worker_parts
    
    def create_workers(
        self,
        worker_positions: List[Tuple[float, float, float]],
        current_hour: int = 12
    ) -> List[go.Scatter3d]:
        """
        Create humanoid worker representations.
        
        Workers are only visible during working hours (9-17).
        
        Args:
            worker_positions: List of (x, y, z) positions
            current_hour: Current hour (0-23)
            
        Returns:
            List of Plotly Scatter3d objects for workers
        """
        workers = []
        
        # Check if it's working hours (9 AM to 5 PM)
        is_working_hours = config.WORK_START_HOUR <= current_hour < config.WORK_END_HOUR
        
        if not worker_positions or not is_working_hours:
            # Return empty workers if not working hours
            return [go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                name='Workers',
                showlegend=True,
                hovertemplate='<extra></extra>'
            )]
        
        for idx, position in enumerate(worker_positions):
            worker_parts = self.create_humanoid_worker(position, worker_id=idx + 1)
            workers.extend(worker_parts)
        
        return workers
    
    def generate_random_worker_positions(
        self,
        num_workers: int
    ) -> List[Tuple[float, float, float]]:
        """
        Generate random positions for workers inside the building.
        
        Args:
            num_workers: Number of workers
            
        Returns:
            List of (x, y, z) positions
        """
        np.random.seed(42)
        positions = []
        for _ in range(num_workers):
            x = np.random.uniform(2, config.BUILDING_WIDTH - 2)
            y = np.random.uniform(2, config.BUILDING_DEPTH - 2)
            z = np.random.uniform(0.5, config.BUILDING_HEIGHT - 1)
            positions.append((x, y, z))
        
        return positions
    
    @staticmethod
    def _compute_dimensions(roof_area: float, tank_capacity: float) -> Dict[str, float]:
        """
        Derive 3D geometry from user-supplied roof area (m^2) and tank
        capacity (L).

        - Building footprint is assumed square: ``side = sqrt(roof_area)``
        - Building height grows mildly with footprint, clamped to sane range
        - Tank is a cylinder with height ~ 2 * radius (volume preserved);
          if that radius would overflow the roof, it is clamped and the
          height is recomputed to keep the same volume.
        """
        roof_area = max(10.0, float(roof_area))
        tank_capacity = max(100.0, float(tank_capacity))

        side = math.sqrt(roof_area)
        bw = side
        bd = side
        bh = max(6.0, min(30.0, side * 0.6))

        volume_m3 = tank_capacity / 1000.0
        r = (volume_m3 / (2.0 * math.pi)) ** (1.0 / 3.0)
        r_max = min(bw, bd) / 2.0 * 0.7
        if r > r_max:
            r = max(0.4, r_max)
            h = volume_m3 / (math.pi * r * r)
        else:
            h = 2.0 * r

        return {
            'bw': bw, 'bd': bd, 'bh': bh,
            'tr': max(0.4, r), 'th': max(0.6, h),
        }

    @staticmethod
    def _building_walls(bw: float, bd: float, bh: float) -> go.Mesh3d:
        """Four opaque walls forming a rectangular box."""
        x = [0, bw, bw, 0, 0, bw, bw, 0]
        y = [0, 0, bd, bd, 0, 0, bd, bd]
        z = [0, 0, 0, 0, bh, bh, bh, bh]
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        return go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            name='Bina',
            color='#A9A9A9', opacity=0.85,
            showlegend=True,
        )

    @staticmethod
    def _flat_roof(bw: float, bd: float, bh: float) -> go.Mesh3d:
        """Flat slab roof at z = bh (acts as rainwater collection plane)."""
        x = [0, bw, bw, 0]
        y = [0, 0, bd, bd]
        z = [bh, bh, bh, bh]
        i = [0, 0]
        j = [1, 2]
        k = [2, 3]
        return go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            name='Çatı',
            color='#8B4513', opacity=0.95,
            showlegend=True,
        )

    @staticmethod
    def _tank_surface(
        bw: float, bd: float, bh: float,
        tr: float, th: float,
        fill_pct: float,
    ) -> go.Surface:
        """Cylindrical tank centered on top of the flat roof."""
        cx = bw / 2.0
        cy = bd / 2.0
        z0 = bh

        theta = np.linspace(0, 2 * np.pi, 22)
        z_range = np.linspace(z0, z0 + th, 14)
        Theta, Z = np.meshgrid(theta, z_range)
        X = cx + tr * np.cos(Theta)
        Y = cy + tr * np.sin(Theta)

        fill_pct = max(0.0, min(100.0, fill_pct))
        fill_z = z0 + th * fill_pct / 100.0
        colors = np.where(Z <= fill_z, 0.0, 1.0)

        return go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=colors,
            colorscale=[[0, '#1E90FF'], [1, '#D3D3D3']],
            cmin=0, cmax=1,
            name='Depo',
            showlegend=True,
            showscale=False,
            hovertemplate=f'Depo dolum: {fill_pct:.1f}%<extra></extra>',
        )

    @staticmethod
    def _rain_scatter(
        bw: float, bd: float, bh: float,
        tr: float, th: float,
        intensity_mm: float,
        frame_index: int,
    ) -> go.Scatter3d:
        """Rain particles falling inside the building footprint."""
        max_particles = config.RAIN_PARTICLE_COUNT_MAX
        n = int(max(0, min(max_particles, (intensity_mm / 50.0) * max_particles)))

        if intensity_mm < 0.1 or n == 0:
            return go.Scatter3d(
                x=[], y=[], z=[], mode='markers',
                marker=dict(size=3, color=config.COLOR_RAIN),
                name='Yağmur',
                showlegend=True,
                hovertemplate='<extra></extra>',
            )

        rng = np.random.RandomState(42)
        bx = rng.uniform(-1.5, bw + 1.5, n)
        by = rng.uniform(-1.5, bd + 1.5, n)

        ceiling = bh + th + 10.0
        phase = (frame_index + np.arange(n)) % 100
        z = ceiling - (phase * 15.0 / 100.0) % ceiling

        opacity = min(0.85, 0.35 + (intensity_mm / 100.0) * 0.5)

        return go.Scatter3d(
            x=bx, y=by, z=z, mode='markers',
            marker=dict(size=3.5, color=config.COLOR_RAIN,
                        opacity=opacity, line=dict(width=0)),
            name='Yağmur',
            showlegend=True,
            hovertemplate=f'Yağış: {intensity_mm:.1f} mm<extra></extra>',
        )

    def create_animated_scene(
        self,
        daily_rain: np.ndarray,
        tank_pcts: np.ndarray,
        labels: List[str],
        roof_area: float,
        tank_capacity: float,
        title: str = "Yağmur Hasadı Animasyonu",
        frame_duration_ms: int = 150,
    ) -> go.Figure:
        """
        Build a Plotly animation of building + tank + rain that is driven by
        real per-day rainfall and per-day tank fill percentages.

        Only the tank surface and rain particles change across frames; the
        building geometry stays fixed to keep animations lightweight.

        Args:
            daily_rain: Array of daily rainfall (mm), length N
            tank_pcts: Array of daily tank fill percentages (0-100), length N
            labels: Human-readable labels (dates) for each frame, length N
            title: Figure title
            frame_duration_ms: Duration of each frame in ms

        Returns:
            Plotly Figure with frames, Play/Pause buttons, and a slider.
        """
        daily_rain = np.asarray(daily_rain, dtype=float)
        tank_pcts = np.asarray(tank_pcts, dtype=float)
        n = int(min(len(daily_rain), len(tank_pcts), len(labels)))
        if n == 0:
            return go.Figure()

        dims = self._compute_dimensions(roof_area, tank_capacity)
        bw, bd, bh = dims['bw'], dims['bd'], dims['bh']
        tr, th = dims['tr'], dims['th']

        fig = go.Figure()
        fig.add_trace(self._building_walls(bw, bd, bh))
        fig.add_trace(self._flat_roof(bw, bd, bh))

        tank_idx = len(fig.data)
        fig.add_trace(self._tank_surface(bw, bd, bh, tr, th, float(tank_pcts[0])))

        rain_idx = len(fig.data)
        fig.add_trace(self._rain_scatter(bw, bd, bh, tr, th,
                                          float(daily_rain[0]), 0))

        frames = []
        for i in range(n):
            tank_trace = self._tank_surface(bw, bd, bh, tr, th, float(tank_pcts[i]))
            rain_trace = self._rain_scatter(bw, bd, bh, tr, th,
                                             float(daily_rain[i]), i)
            frames.append(go.Frame(
                data=[tank_trace, rain_trace],
                traces=[tank_idx, rain_idx],
                name=str(labels[i])
            ))
        fig.frames = frames

        slider_steps = [
            dict(
                method="animate",
                args=[[str(labels[i])], dict(
                    mode="immediate",
                    frame=dict(duration=frame_duration_ms, redraw=True),
                    transition=dict(duration=0)
                )],
                label=str(labels[i])
            )
            for i in range(n)
        ]

        play_button = dict(
            label="Oynat",
            method="animate",
            args=[None, dict(
                frame=dict(duration=frame_duration_ms, redraw=True),
                fromcurrent=True,
                transition=dict(duration=0)
            )]
        )
        pause_button = dict(
            label="Durdur",
            method="animate",
            args=[[None], dict(
                frame=dict(duration=0, redraw=False),
                mode="immediate",
                transition=dict(duration=0)
            )]
        )

        x_pad = max(3.0, bw * 0.15)
        y_pad = max(3.0, bd * 0.15)
        z_top = bh + th + 10.0

        fig.update_layout(
            title={
                'text': title,
                'x': 0.5, 'xanchor': 'center',
                'font': dict(size=16),
            },
            scene=dict(
                xaxis=dict(
                    title='X (m)',
                    range=[-x_pad, bw + x_pad],
                    backgroundcolor='#E0E0E0',
                ),
                yaxis=dict(
                    title='Y (m)',
                    range=[-y_pad, bd + y_pad],
                    backgroundcolor='#E0E0E0',
                ),
                zaxis=dict(
                    title='Yükseklik (m)',
                    range=[0, z_top],
                    backgroundcolor='#87CEEB',
                ),
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.1),
                    center=dict(x=0, y=0, z=-0.1),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode='data',
                bgcolor='#E8F4F8',
            ),
            width=1050,
            height=780,
            margin=dict(l=0, r=0, b=0, t=90),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top', y=1.12,
                xanchor='center', x=0.5,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#333', borderwidth=1,
                font=dict(size=14, color='#111', family='Arial'),
                itemsizing='constant',
            ),
            paper_bgcolor='#F5F5F5',
            font=dict(family='Arial', size=12),
            updatemenus=[dict(
                type='buttons',
                direction='left',
                x=0.02, y=1.18,
                xanchor='left', yanchor='top',
                pad=dict(r=10, t=10),
                showactive=False,
                bgcolor='#FFFFFF',
                bordercolor='#333',
                borderwidth=1,
                font=dict(size=13),
                buttons=[play_button, pause_button],
            )],
            sliders=[dict(
                active=0,
                x=0.05, y=0,
                len=0.9,
                pad=dict(b=10, t=40),
                currentvalue=dict(
                    prefix='Tarih: ',
                    visible=True,
                    xanchor='right',
                    font=dict(size=13),
                ),
                font=dict(size=11),
                steps=slider_steps,
            )],
        )

        return fig

    def create_full_scene(
        self,
        tank_level_percentage: float = 50,
        rain_intensity: float = 0,
        num_workers: int = 10,
        current_hour: int = 12,
        frame_index: int = 0
    ) -> go.Figure:
        """
        Create complete 3D scene with all enhanced components.
        
        Args:
            tank_level_percentage: Tank fill level (0-100)
            rain_intensity: Current rainfall (0-100 mm)
            num_workers: Number of workers to display
            current_hour: Current hour (0-23)
            frame_index: Animation frame for rain
            
        Returns:
            Plotly Figure object
        """
        # Create figure
        self.fig = go.Figure()
        
        # Add realistic building components
        building_parts = self.create_realistic_building()
        for part in building_parts:
            self.fig.add_trace(part)
        
        # Add tank components
        self.fig.add_trace(self.create_realistic_tank(tank_level_percentage))
        self.fig.add_trace(self.create_tank_base())
        
        # Add animated rain
        self.fig.add_trace(self.create_animated_rain_particles(
            rain_intensity, frame_index
        ))
        
        # Add workers (only during working hours)
        worker_positions = self.generate_random_worker_positions(min(num_workers, 20))
        worker_traces = self.create_workers(worker_positions, current_hour)
        for worker_trace in worker_traces:
            self.fig.add_trace(worker_trace)
        
        # Update layout with better camera and scene settings
        self.fig.update_layout(
            title={
                'text': f'Yağmur Hasadı Sistemi - 3D Görünüm (Gün: Dinamik, Saat: {current_hour:02d}:00)',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis=dict(
                    title='X (meter)',
                    range=[0, config.BUILDING_WIDTH + 12],
                    backgroundcolor='#E0E0E0'
                ),
                yaxis=dict(
                    title='Y (meter)',
                    range=[-2, config.BUILDING_DEPTH + 2],
                    backgroundcolor='#E0E0E0'
                ),
                zaxis=dict(
                    title='Yükseklik (meter)',
                    range=[0, config.BUILDING_HEIGHT + 12],
                    backgroundcolor='#87CEEB'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube',
                bgcolor='#E8F4F8'  # Light cyan background
            ),
            width=1000,
            height=750,
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#000000',
                borderwidth=1
            ),
            hovermode='closest',
            paper_bgcolor='#F5F5F5',
            font=dict(family='Arial', size=11)
        )
        
        return self.fig


class TimeSeriesGraphs:
    """
    Creates 2D time-series visualization graphs with Turkish labels.
    """
    
    @staticmethod
    def create_tank_level_graph(days: List[int], levels: List[float], capacity: float) -> go.Figure:
        """
        Create tank level over time graph with Turkish labels.
        
        Args:
            days: List of day numbers
            levels: List of tank levels
            capacity: Tank capacity for reference
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days,
            y=levels,
            mode='lines',
            name='Depo Seviyesi',
            fill='tozeroy',
            line=dict(color='#1E90FF', width=2),
            fillcolor='rgba(30, 144, 255, 0.3)',
            hovertemplate='Gün: %{x}<br>Seviye: %{y:,.0f} L<extra></extra>'
        ))
        
        # Add capacity line
        fig.add_hline(
            y=capacity,
            line_dash="dash",
            line_color="red",
            annotation_text="Kapasite",
            annotation_position="right"
        )
        
        fig.update_layout(
            title='Zaman İçinde Depo Seviyesi',
            xaxis_title='Gün',
            yaxis_title='Seviye (Litre)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_rainfall_graph(days: List[int], rainfall: List[float]) -> go.Figure:
        """
        Create daily rainfall visualization with Turkish labels.
        
        Args:
            days: List of day numbers
            rainfall: List of daily rainfall amounts
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=days,
            y=rainfall,
            name='Günlük Yağış',
            marker_color='#87CEEB',
            hovertemplate='Gün: %{x}<br>Yağış: %{y:.1f} mm<extra></extra>'
        ))
        
        fig.update_layout(
            title='Günlük Yağış',
            xaxis_title='Gün',
            yaxis_title='Yağış (mm)',
            hovermode='x',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_consumption_vs_supply_graph(
        days: List[int],
        inflow: List[float],
        outflow: List[float]
    ) -> go.Figure:
        """
        Create consumption vs supply comparison graph with Turkish labels.
        
        Args:
            days: List of day numbers
            inflow: List of collected water amounts
            outflow: List of consumed water amounts
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days,
            y=inflow,
            mode='lines',
            name='Toplanan Su',
            line=dict(color='#32CD32', width=2),
            hovertemplate='Gün: %{x}<br>Toplanan: %{y:,.0f} L<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=days,
            y=outflow,
            mode='lines',
            name='Tüketilen Su',
            line=dict(color='#FF6347', width=2),
            hovertemplate='Gün: %{x}<br>Tüketilen: %{y:,.0f} L<extra></extra>'
        ))
        
        fig.update_layout(
            title='Su Arzı vs Tüketim',
            xaxis_title='Gün',
            yaxis_title='Su (Litre)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_monthly_summary_bar(months: List[str], values: List[float], title: str) -> go.Figure:
        """
        Create monthly summary bar chart.
        
        Args:
            months: List of month names
            values: List of values for each month
            title: Chart title (in Turkish)
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=months,
            y=values,
            marker_color='#4682B4',
            hovertemplate='%{x}<br>Değer: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Ay',
            yaxis_title='Değer',
            template='plotly_white'
        )
        
        return fig


# ---------------------------------------------------------------------------
# Map / PyDeck helpers
# ---------------------------------------------------------------------------


def build_scaled_building_svg(
    roof_area_m2: float,
    tank_capacity_l: float,
    fill_pct: float = 0.0,
) -> Tuple[str, int, int]:
    """
    Build an isometric SVG of a building with a rooftop cylindrical tank,
    scaled by the user's roof area (m^2) and tank capacity (L).

    The SVG is returned as a raw string suitable for embedding in a Folium
    ``DivIcon`` so the marker itself visually reflects the simulated
    building instead of a generic pin.

    Args:
        roof_area_m2: Roof area in m^2 (clamped to >= 10).
        tank_capacity_l: Tank capacity in liters (clamped to >= 100).
        fill_pct: Optional 0..100 fill indicator drawn inside the tank.

    Returns:
        Tuple of ``(svg_string, icon_width_px, icon_height_px)``.
    """
    roof_area = max(10.0, float(roof_area_m2))
    tank_capacity = max(100.0, float(tank_capacity_l))
    fill = max(0.0, min(100.0, float(fill_pct)))

    # Footprint side in pixels (clamped for UI).
    side = max(40.0, min(120.0, math.sqrt(roof_area) * 1.4))
    height = max(side * 0.55, 40.0)

    # Tank radius/height in pixels (independent of 3D scene logic).
    volume_m3 = tank_capacity / 1000.0
    tr_world = (volume_m3 / (2.0 * math.pi)) ** (1.0 / 3.0)
    tr = max(6.0, min(side * 0.32, tr_world * 6.0))
    th = max(tr * 1.6, 18.0)

    # Isometric projection offsets.
    ox = side * 0.5
    oy = side * 0.28

    # Canvas sizing with padding for shadow + tank.
    pad = 12
    canvas_w = int(side + ox + 2 * pad)
    canvas_h = int(height + oy + th + 2 * pad + 10)

    # Base building anchor so the bottom-front corner sits near bottom center.
    base_x = pad
    base_y = canvas_h - pad - 10  # leave room for ground shadow

    # Building corners (front-bottom-left at (base_x, base_y)).
    p_fbl = (base_x, base_y)
    p_fbr = (base_x + side, base_y)
    p_bbr = (base_x + side + ox, base_y - oy)
    p_bbl = (base_x + ox, base_y - oy)

    p_ftl = (p_fbl[0], p_fbl[1] - height)
    p_ftr = (p_fbr[0], p_fbr[1] - height)
    p_btr = (p_bbr[0], p_bbr[1] - height)
    p_btl = (p_bbl[0], p_bbl[1] - height)

    def _pt(p):
        return f"{p[0]:.1f},{p[1]:.1f}"

    front_face = f"{_pt(p_fbl)} {_pt(p_fbr)} {_pt(p_ftr)} {_pt(p_ftl)}"
    side_face = f"{_pt(p_fbr)} {_pt(p_bbr)} {_pt(p_btr)} {_pt(p_ftr)}"
    roof_face = f"{_pt(p_ftl)} {_pt(p_ftr)} {_pt(p_btr)} {_pt(p_btl)}"

    # Rooftop tank: centered on the roof quadrilateral.
    roof_cx = (p_ftl[0] + p_btr[0]) / 2
    roof_cy = (p_ftl[1] + p_btr[1]) / 2
    tank_top_y = roof_cy - th
    tank_bottom_y = roof_cy

    # Elliptical caps.
    ry = tr * 0.35
    # Fill level inside tank.
    fill_y = tank_bottom_y - th * (fill / 100.0)

    # Ground shadow ellipse.
    sh_cx = base_x + side / 2 + ox / 2
    sh_cy = base_y + 6
    sh_rx = side * 0.72
    sh_ry = 6

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}" style="overflow: visible; cursor: grab;">
  <defs>
    <linearGradient id="wallFront" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#CFD8DC"/>
      <stop offset="100%" stop-color="#90A4AE"/>
    </linearGradient>
    <linearGradient id="wallSide" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#78909C"/>
      <stop offset="100%" stop-color="#455A64"/>
    </linearGradient>
    <linearGradient id="roofGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ECEFF1"/>
      <stop offset="100%" stop-color="#B0BEC5"/>
    </linearGradient>
    <linearGradient id="tankGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#64B5F6"/>
      <stop offset="50%" stop-color="#1E88E5"/>
      <stop offset="100%" stop-color="#0D47A1"/>
    </linearGradient>
  </defs>
  <ellipse cx="{sh_cx:.1f}" cy="{sh_cy:.1f}" rx="{sh_rx:.1f}" ry="{sh_ry:.1f}" fill="rgba(0,0,0,0.25)"/>
  <polygon points="{front_face}" fill="url(#wallFront)" stroke="#37474F" stroke-width="1.2"/>
  <polygon points="{side_face}" fill="url(#wallSide)" stroke="#263238" stroke-width="1.2"/>
  <polygon points="{roof_face}" fill="url(#roofGrad)" stroke="#37474F" stroke-width="1.2"/>
  <!-- Tank body: back ellipse (top), rectangle, front ellipse -->
  <rect x="{roof_cx - tr:.1f}" y="{tank_top_y:.1f}" width="{2*tr:.1f}" height="{th:.1f}" fill="#90CAF9" stroke="#0D47A1" stroke-width="1.2"/>
  <rect x="{roof_cx - tr:.1f}" y="{fill_y:.1f}" width="{2*tr:.1f}" height="{tank_bottom_y - fill_y:.1f}" fill="url(#tankGrad)" opacity="0.85"/>
  <ellipse cx="{roof_cx:.1f}" cy="{tank_top_y:.1f}" rx="{tr:.1f}" ry="{ry:.1f}" fill="#BBDEFB" stroke="#0D47A1" stroke-width="1.2"/>
  <ellipse cx="{roof_cx:.1f}" cy="{tank_bottom_y:.1f}" rx="{tr:.1f}" ry="{ry:.1f}" fill="#1976D2" stroke="#0D47A1" stroke-width="1.2"/>
</svg>
""".strip()

    return svg, canvas_w, canvas_h


def build_pydeck_overlay(
    lat: float,
    lon: float,
    rain_mm_today: float,
    tank_pct_today: float,
    roof_area_m2: float,
    tank_capacity_l: float,
    frame_idx: int = 0,
    seed: int = 42,
):
    """
    Construct a PyDeck ``Deck`` rendering:

    - an extruded square polygon representing the building footprint,
    - a vertical column for the storage tank whose height reflects current
      fill percentage,
    - a scatter of rain particles around the building whose density scales
      with ``rain_mm_today``.

    The overlay is intended to be layered on top of a map via
    ``st.pydeck_chart``.
    """
    import pydeck as pdk

    lat = float(lat)
    lon = float(lon)
    roof_area = max(10.0, float(roof_area_m2))
    tank_capacity = max(100.0, float(tank_capacity_l))
    rain_mm = max(0.0, float(rain_mm_today))
    pct = max(0.0, min(100.0, float(tank_pct_today)))

    # Real-world dimensions.
    side_m = math.sqrt(roof_area)
    bh_m = max(6.0, min(30.0, side_m * 0.6))
    volume_m3 = tank_capacity / 1000.0
    # Fiziksel silindir ölçüsü (referans).
    tr_m = max(0.4, (volume_m3 / (2.0 * math.pi)) ** (1.0 / 3.0))
    tr_max = side_m * 0.35
    if tr_m > tr_max:
        tr_m = tr_max
    # Görsel tank: doluluk farkı net görünsün diye bina yüksekliğine yakın
    # ölçeklenir ve daha tombul bir yarıçap kullanılır. Bu, gerçek fizikten
    # ziyade görsel kavrayış amaçlıdır.
    th_m = max(6.0, min(18.0, bh_m * 0.9))
    tank_radius = max(1.5, min(4.0, side_m * 0.22))

    # 1 meter in degrees (approximate).
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * max(0.1, math.cos(math.radians(lat))))

    half_side_lat = (side_m / 2.0) * deg_per_m_lat
    half_side_lon = (side_m / 2.0) * deg_per_m_lon

    building_polygon = [
        [lon - half_side_lon, lat - half_side_lat],
        [lon + half_side_lon, lat - half_side_lat],
        [lon + half_side_lon, lat + half_side_lat],
        [lon - half_side_lon, lat + half_side_lat],
    ]

    building_layer = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": building_polygon, "height": bh_m, "name": "Bina"}],
        get_polygon="polygon",
        get_elevation="height",
        extruded=True,
        wireframe=True,
        get_fill_color=[160, 170, 180, 220],
        get_line_color=[60, 70, 80, 255],
        pickable=True,
    )

    # Tank column: ColumnLayer başlangıç yüksekliği olmadığı (zeminden çıkar)
    # için binanın içinde görünmez olmaması adına tank bina footprint'inin
    # hemen doğusuna — tankın çapı kadar — offsetlenir. Sadece dolu iç
    # sütun çizilir; yarı saydam bir "silüet" sütun depth-buffer sorunuyla
    # iç sütunu gizlediği için eklenmez. Kapasite bağlamı üstteki TextLayer
    # etiketiyle sağlanır.
    tank_lon = lon + half_side_lon + 2.2 * tank_radius * deg_per_m_lon
    tank_lat = lat
    visible_height = max(0.3, th_m * (pct / 100.0))
    outline_height = th_m
    # Doluluk arttıkça belirgin koyu laciverte kayan opak mavi.
    tank_fill_color = [
        max(5, 20 - int(pct * 0.1)),
        max(60, 120 - int(pct * 0.3)),
        int(min(255, 180 + pct * 0.6)),
        245,
    ]
    tank_layer = pdk.Layer(
        "ColumnLayer",
        data=[{
            "position": [tank_lon, tank_lat],
            "height": visible_height,
            "pct": pct,
            "name": f"Depo {pct:.0f}%",
        }],
        get_position="position",
        get_elevation="height",
        elevation_scale=1,
        radius=tank_radius * 0.92,
        get_fill_color=tank_fill_color,
        pickable=True,
        extruded=True,
    )

    # Rain particles: density scales with intensity, capped for perf.
    n_particles = int(min(400, max(0, rain_mm * 40)))
    rng = np.random.RandomState(seed + int(frame_idx) * 7)
    rain_data = []
    if n_particles > 0:
        lat_spread = 60 * deg_per_m_lat
        lon_spread = 60 * deg_per_m_lon
        for _ in range(n_particles):
            dlat = rng.uniform(-lat_spread, lat_spread)
            dlon = rng.uniform(-lon_spread, lon_spread)
            phase = rng.uniform(0, 1)
            # Let elevation cycle per frame to simulate falling.
            elev = ((phase * 100 - (frame_idx * 6)) % 80) + 4
            rain_data.append({
                "position": [lon + dlon, lat + dlat, float(elev)],
            })

    rain_layer = pdk.Layer(
        "ScatterplotLayer",
        data=rain_data,
        get_position="position",
        get_radius=0.6,
        get_fill_color=[100, 180, 255, 200],
        radius_min_pixels=1,
        radius_max_pixels=3,
    )

    # Doluluk yüzde etiketi (tankın tepesinde, kameraya dönük).
    label_layer = pdk.Layer(
        "TextLayer",
        data=[{
            "position": [tank_lon, tank_lat, outline_height + 1.5],
            "label": f"{pct:.0f}%",
        }],
        get_position="position",
        get_text="label",
        get_size=20,
        get_color=[13, 71, 161, 255],
        get_alignment_baseline="'bottom'",
        get_text_anchor="'middle'",
        billboard=True,
    )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=17.5,
        pitch=55,
        bearing=30,
    )

    deck = pdk.Deck(
        layers=[building_layer, tank_layer, label_layer, rain_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"text": "{name}"},
    )
    return deck
