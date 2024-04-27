import pygame
import time
import math
from utils import scale_image, blit_rotate_center

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

mask_surface = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
mask = pygame.mask.from_surface(mask_surface)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

FPS = 60

#####

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

beam_surface = pygame.Surface((500, 500), pygame.SRCALPHA)

mask_fx = pygame.mask.from_surface(pygame.transform.flip(mask_surface, True, False))
mask_fy = pygame.mask.from_surface(pygame.transform.flip(mask_surface, False, True))
mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(mask_surface, True, True))
flipped_masks = [[mask, mask_fy], [mask_fx, mask_fx_fy]]


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()


def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)
    pygame.display.update()


def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward()
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backward()

    if not moved:
        player_car.reduce_speed()

def draw_beam(surface, angle, pos):
    c = math.cos(math.radians(angle))
    s = math.sin(math.radians(angle))

    flip_x = c < 0
    flip_y = s < 0
    filpped_mask = flipped_masks[flip_x][flip_y]
    
    # compute beam final point
    x_dest = WIDTH * abs(c)
    y_dest = HEIGHT * abs(s)

    beam_surface.fill((0, 0, 0, 0))

    # draw a single beam to the beam surface based on computed final point
    pygame.draw.line(beam_surface, BLUE, (0, 0), (x_dest, y_dest))
    beam_mask = pygame.mask.from_surface(beam_surface)

    # find overlap between "global mask" and current beam mask
    offset_x = WIDTH-1-pos[0] if flip_x else pos[0]
    offset_y = HEIGHT-1-pos[1] if flip_y else pos[1]
    hit = filpped_mask.overlap(beam_mask, (offset_x, offset_y))
    if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
        hx = WIDTH-1 - hit[0] if flip_x else hit[0]
        hy = HEIGHT-1 - hit[1] if flip_y else hit[1]
        hit_pos = (hx, hy)

        pygame.draw.line(surface, BLUE, pos, hit_pos)
        pygame.draw.circle(surface, GREEN, hit_pos, 3)

    pygame.display.update()


run = True
clock = pygame.time.Clock()
images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
          (FINISH, FINISH_POSITION), (mask_surface, (0, 0))]
# images = [(mask_surface, (0, 0))]
player_car = PlayerCar(8, 8)

while run:
    clock.tick(FPS)

    draw(WIN, images, player_car)

    beam_origin = player_car.x + RED_CAR.get_width() // 2, player_car.y + RED_CAR.get_height() // 2

    for angle in range(0, 359, 30):
        draw_beam(WIN, angle, beam_origin)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    move_player(player_car)

    if player_car.collide(mask) != None:
        player_car.bounce()

    finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    if finish_poi_collide != None:
        if finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            player_car.reset()
            print("finish")


pygame.quit()
