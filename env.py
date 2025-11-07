import os
import numpy as np
import random
import pygame
import config as cf


class FlappyBirdEnv:
    def __init__(self, difficulty="normal", render_mode=False):
        # ===== Config values =====
        self.ASSET_DIR_NAME = cf.ASSET_DIR_NAME
        self.SCREEN_WIDTH = cf.SCREEN_WIDTH
        self.SCREEN_HEIGHT = cf.SCREEN_HEIGHT
        self.FPS = cf.FPS
        self.render_mode = render_mode

        # ===== Named constants =====
        self.INIT_PIPE_OFFSET = cf.INIT_PIPE_OFFSET 
        self.MIN_GAP_Y = cf.MIN_GAP_Y
        self.GROUND_HEIGHT = cf.GROUND_HEIGHT 

        # ===== Difficulty presets =====
        presets = {
            "easy": {"PIPE_GAP": 220, "PIPE_SPACING": 300, "SCROLL_SPEED": 2},
            "normal": {"PIPE_GAP": 180, "PIPE_SPACING": 280, "SCROLL_SPEED": 3},
            "hard": {"PIPE_GAP": 150, "PIPE_SPACING": 260, "SCROLL_SPEED": 3},
            "extreme": {"PIPE_GAP": 130, "PIPE_SPACING": 240, "SCROLL_SPEED": 4}
        }
        if difficulty not in presets:
            raise ValueError("Difficulty must be 'easy', 'normal', or 'hard'.")

        self.PIPE_GAP = presets[difficulty]["PIPE_GAP"]
        self.PIPE_SPACING = presets[difficulty]["PIPE_SPACING"]
        self.SCROLL_SPEED = presets[difficulty]["SCROLL_SPEED"]

        # ===== Physics =====
        self.GRAVITY = cf.GRAVITY
        self.FLAP_VEL = cf.FLAP_VEL
        self.MAX_VEL = cf.MAX_VEL

        # ===== Reward Hyperparameters =====
        # Basic reward values
        self.LIVING_REWARD = cf.LIVING_REWARD
        self.SCORE_REWARD = cf.SCORE_REWARD
        self.DEATH_PENALTY = cf.DEATH_PENALTY
        # Reward shaping weights      
        self.VERTICAL_WEIGHT = cf.VERTICAL_WEIGHT
        self.VELOCITY_WEIGHT = cf.VELOCITY_WEIGHT
        self.CENTER_BONUS_MULT = cf.CENTER_BONUS_MULT
        self.APPROACHING_THRESHOLD = cf.APPROACHING_THRESHOLD
        self.APPROACHING_MULTIPLIER = cf.APPROACHING_MULTIPLIER

        # ===== Color theme =====
        self.colors = {
            "bg": (135, 206, 235),
            "pipe": (34, 139, 34),
            "ground": (222, 184, 135),
            "bird": (255, 215, 0),
            "ui": (0, 0, 0)
        }

        # ===== Pygame init =====
        pygame.init()
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird RL")
            self.clock = pygame.time.Clock()

        # ===== Load assets =====
        base_dir = os.path.dirname(__file__)
        asset_dir = os.path.join(base_dir, self.ASSET_DIR_NAME)
        self._asset_dir = asset_dir
        self._load_assets_safe()

        # ===== State initialization =====
        self.reset()

    def _load_assets_safe(self):
        """Try to load images; fallback to shapes if failed."""
        try:
            bg_path = os.path.join(self._asset_dir, "bg.png")
            bird_up_path = os.path.join(self._asset_dir, "yellowbird-upflap.png")
            bird_mid_path = os.path.join(self._asset_dir, "yellowbird-midflap.png")
            bird_down_path = os.path.join(self._asset_dir, "yellowbird-downflap.png")
            base_path = os.path.join(self._asset_dir, "base.png")
            pipe_path = os.path.join(self._asset_dir, "pipe-green.png")

            self.bg_img = pygame.image.load(bg_path).convert() if self.render_mode else pygame.image.load(bg_path)
            self.bg_img = pygame.transform.scale(self.bg_img, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

            def load_img(path):
                return pygame.image.load(path).convert_alpha() if self.render_mode else pygame.image.load(path)

            self.bird_frames = [load_img(bird_up_path), load_img(bird_mid_path), load_img(bird_down_path)]
            self.base_img = load_img(base_path)
            base_h = self.base_img.get_height()
            self.base_img = pygame.transform.scale(self.base_img, (self.SCREEN_WIDTH, base_h))
            self.pipe_img = load_img(pipe_path)
            self.pipe_width = self.pipe_img.get_width()
            self.use_image = True
        except Exception as e:
            print(f"Warning: Assets not found. Using simple shapes. {e}")
            self.use_image = False
            self.bird_frames = None
            self.base_img = None
            self.pipe_img = None
            self.pipe_width = cf.PIPE_WIDTH

    def reset(self):
        """Reset environment to initial state."""
        # Bird
        self.bird_x = cf.BIRD_X_POS
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.bird_radius = cf.BIRD_RADIUS

        # Animation
        self.bird_frame = 0
        self.anim_timer = 0
        self.ANIM_FREQ = cf.ANIM_FREQ

        # Scroller
        self.bg_x = 0
        self.base_x = 0

        # Pipes
        self.pipes = []
        start_x = self.SCREEN_WIDTH + self.INIT_PIPE_OFFSET
        for i in range(3):
            gap_y = random.randint(self.MIN_GAP_Y, self.SCREEN_HEIGHT - self.MIN_GAP_Y - self.PIPE_GAP - self.GROUND_HEIGHT)
            self.pipes.append([start_x + i * self.PIPE_SPACING, gap_y])

        self.done = False
        self.score = 0
        self.scored_pipes = set()

        return self._get_state()

    def step(self, action):
        """
        Main game loop step.
        
        Args:
            action: 0 = do nothing, 1 = flap
            
        Returns:
            state, reward, done, info
        """
        # ===== Apply action =====
        if action == 1:
            self.bird_vel = self.FLAP_VEL

        # ===== Physics update =====
        self.bird_vel += self.GRAVITY
        self.bird_vel = max(-self.MAX_VEL, min(self.MAX_VEL, self.bird_vel))
        self.bird_y += self.bird_vel

        # ===== Animation =====
        self.anim_timer += 1
        if self.anim_timer >= self.ANIM_FREQ:
            self.anim_timer = 0
            self.bird_frame = (self.bird_frame + 1) % 3

        # ===== Background scroll =====
        self.base_x -= self.SCROLL_SPEED
        if self.base_x <= -self.SCREEN_WIDTH:
            self.base_x = 0
        self.bg_x -= self.SCROLL_SPEED / 2
        if self.bg_x <= -self.SCREEN_WIDTH:
            self.bg_x = 0

        # ===== Move pipes =====
        for i in range(len(self.pipes)):
            self.pipes[i][0] -= self.SCROLL_SPEED

        # Remove old pipe, add new pipe
        if len(self.pipes) > 0 and (self.pipes[0][0] + self.pipe_width) < 0:
            popped = self.pipes.pop(0)
            self.scored_pipes.discard(id(popped))

            new_x = self.pipes[-1][0] + self.PIPE_SPACING
            new_gap_y = random.randint(self.MIN_GAP_Y, self.SCREEN_HEIGHT - self.MIN_GAP_Y - self.PIPE_GAP - self.GROUND_HEIGHT)
            self.pipes.append([new_x, new_gap_y])

        # ===== Collision check =====
        if self._check_collision():
            self.done = True
            reward = self._reward_death()
            return self._get_state(), reward, self.done, {}

        # ===== Scoring check =====
        just_scored = False
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            if (pipe_x + self.pipe_width) < self.bird_x and id(pipe) not in self.scored_pipes:
                if gap_y < self.bird_y < gap_y + self.PIPE_GAP:
                    self.score += 1
                    just_scored = True
                self.scored_pipes.add(id(pipe))

        # ===== Calculate reward =====
        dy_norm, vel_norm, dx_norm = self._get_normalized_values()
        
        if just_scored:
            reward = self._reward_score(dy_norm)
        else:
            reward = self._reward_alive(dy_norm, vel_norm, dx_norm)

        return self._get_state(), reward, self.done, {}

    # =========================================================================
    # REWARD FUNCTIONS 
    # =========================================================================

    def _reward_alive(self, dy_norm, vel_norm, dx_norm):
        """
        Reward when the bird is alive.
        
        Components:
          - Living bonus 
          - Vertical alignment penalty
          - Velocity penalty
        """
        reward = self.LIVING_REWARD
        reward += self._reward_vertical_alignment(dy_norm, dx_norm)
        reward += self._reward_velocity_penalty(vel_norm)
        return reward

    def _reward_score(self, dy_norm):
        """
        Reward when passing a pipe.

        Components:
          - Base score reward
        """
        base = self.SCORE_REWARD
        center_bonus = (1.0 - abs(dy_norm)) * self.CENTER_BONUS_MULT
        center_bonus = max(0, center_bonus)
        return base + center_bonus

    def _reward_death(self):
        """Penalty on death."""
        return self.DEATH_PENALTY

    def _reward_vertical_alignment(self, dy_norm, dx_norm):
        """
        Penalty when not align with the gap.
        Penalty is doubled when approaching the pipe (dx_norm < threshold).
        """
        penalty = -abs(dy_norm) * self.VERTICAL_WEIGHT
        
        if dx_norm < self.APPROACHING_THRESHOLD:
            penalty *= self.APPROACHING_MULTIPLIER
        
        return penalty

    def _reward_velocity_penalty(self, vel_norm):
        """Penalty cho velocity cao - khuyến khích stable flight."""
        return -abs(vel_norm) * self.VELOCITY_WEIGHT

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def _get_normalized_values(self):
        """
        Normalized values for reward shaping:
            - dy_norm: vertical distance to gap center / PIPE_GAP
            - vel_norm: bird velocity / MAX_VEL
            - dx_norm: horizontal distance to next pipe / SCREEN_WIDTH
        
        Returns:
            dy_norm, vel_norm, dx_norm
        """
        next_pipe_x, next_gap_y = self._get_next_pipe()
        gap_center = next_gap_y + self.PIPE_GAP / 2
        
        dy_norm = (gap_center - self.bird_y) / float(self.PIPE_GAP)
        vel_norm = self.bird_vel / float(self.MAX_VEL)
        dx_norm = (next_pipe_x - self.bird_x) / float(self.SCREEN_WIDTH)
        
        return dy_norm, vel_norm, dx_norm

    def _get_next_pipe(self):
        """Get the next pipe in front of the bird."""
        for pipe_x, gap_y in self.pipes:
            if pipe_x + self.pipe_width >= self.bird_x:
                return pipe_x, gap_y
        return self.pipes[0]

    def _check_collision(self):
        """
        Check collision with ground/ceiling/pipes.
        
        Returns:
            True if collision detected
        """
        bird_rect = pygame.Rect(0, 0, self.bird_radius * 2, self.bird_radius * 2)
        bird_rect.center = (int(self.bird_x), int(self.bird_y))
        bird_hitbox = bird_rect.inflate(-6, -6)

        GROUND_Y = self.SCREEN_HEIGHT - self.GROUND_HEIGHT

        # Ground / ceiling
        if self.bird_y - self.bird_radius <= 0 or self.bird_y + self.bird_radius >= GROUND_Y:
            return True

        # Pipes
        for pipe_x, gap_y in self.pipes:
            pipe_top_rect = pygame.Rect(int(pipe_x), 0, self.pipe_width, int(gap_y))
            pipe_bot_rect = pygame.Rect(int(pipe_x), int(gap_y + self.PIPE_GAP), 
                                       self.pipe_width,
                                       self.SCREEN_HEIGHT - int(gap_y + self.PIPE_GAP))
            if bird_hitbox.colliderect(pipe_top_rect) or bird_hitbox.colliderect(pipe_bot_rect):
                return True

        return False

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self):
        """Render game state (only if render_mode=True)."""
        if not self.render_mode:
            return

        # Background
        if self.use_image and hasattr(self, "bg_img") and self.bg_img is not None:
            self.screen.blit(self.bg_img, (self.bg_x, 0))
            self.screen.blit(self.bg_img, (self.bg_x + self.SCREEN_WIDTH, 0))
        else:
            self.screen.fill(self.colors["bg"])

        # Pipes
        for pipe_x, gap_y in self.pipes:
            if self.use_image and self.pipe_img is not None:
                pipe_top_img = pygame.transform.flip(self.pipe_img, False, True)
                pipe_top_rect = pipe_top_img.get_rect(midbottom=(pipe_x + self.pipe_width / 2, gap_y))
                pipe_bot_rect = self.pipe_img.get_rect(midtop=(pipe_x + self.pipe_width / 2,
                                                               gap_y + self.PIPE_GAP))
                self.screen.blit(pipe_top_img, pipe_top_rect)
                self.screen.blit(self.pipe_img, pipe_bot_rect)
            else:
                pipe_top_rect = pygame.Rect(int(pipe_x), 0, self.pipe_width, int(gap_y))
                pipe_bot_rect = pygame.Rect(int(pipe_x), int(gap_y + self.PIPE_GAP), self.pipe_width,
                                            self.SCREEN_HEIGHT - int(gap_y + self.PIPE_GAP))
                pygame.draw.rect(self.screen, self.colors["pipe"], pipe_top_rect)
                pygame.draw.rect(self.screen, self.colors["pipe"], pipe_bot_rect)

        # Base (ground)
        GROUND_Y = self.SCREEN_HEIGHT - self.GROUND_HEIGHT
        if self.use_image and self.base_img is not None:
            self.screen.blit(self.base_img, (self.base_x, GROUND_Y))
            self.screen.blit(self.base_img, (self.base_x + self.SCREEN_WIDTH, GROUND_Y))
        else:
            pygame.draw.rect(self.screen, self.colors["ground"], 
                           (0, GROUND_Y, self.SCREEN_WIDTH, self.GROUND_HEIGHT))

        # Bird
        if self.use_image and self.bird_frames is not None:
            bird_img = self.bird_frames[self.bird_frame]
            if (self.anim_timer % 3) == 0:
                rotated = pygame.transform.rotate(bird_img, -self.bird_vel * 3)
            else:
                rotated = bird_img
            bird_rect = rotated.get_rect(center=(int(self.bird_x), int(self.bird_y)))
            self.screen.blit(rotated, bird_rect)
        else:
            pygame.draw.circle(self.screen, self.colors["bird"], 
                             (int(self.bird_x), int(self.bird_y)), self.bird_radius)

        # UI text
        font = pygame.font.SysFont("Fixedsys", 28)
        score_img = font.render(f"Score: {self.score}", True, self.colors["ui"])
        self.screen.blit(score_img, (10, 10))
        hint_img = font.render("Press ESC to quit", True, self.colors["ui"])
        self.screen.blit(hint_img, (10, 40))

        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        """Clean up pygame."""
        if self.render_mode:
            pygame.quit()

    def _get_state(self):
        """
        Return state as 4-dim vector:
         [dy_norm, vel_norm, pipe_dist_norm, gap_y_norm]
         
         gap_y_norm: normalized - top screen to bottom of top pipe
        """
        next_pipe_x, next_gap_y = self._get_next_pipe()
        gap_center = next_gap_y + self.PIPE_GAP / 2
        
        dy_norm = (gap_center - self.bird_y) / float(self.PIPE_GAP)
        vel_norm = self.bird_vel / float(self.MAX_VEL)
        pipe_dist_norm = (next_pipe_x - self.bird_x) / float(self.SCREEN_WIDTH)
        gap_y_norm = next_gap_y / float(self.SCREEN_HEIGHT - self.GROUND_HEIGHT) 

        return np.array([dy_norm, vel_norm, pipe_dist_norm, gap_y_norm], dtype=np.float32)