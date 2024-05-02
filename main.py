import pygame
import time
import math
from utils import scale_image, blit_rotate_center, euclidean_distance
from model import Network

pygame.font.init()

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
        self.angle = 180
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
        self.angle = 180
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (50, 200)

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

def move_cpu(player_car, network, state):
    # actions = [
    #             (0,0), (0,1), (0, 2), 
    #             (1,0), (1,1), (1, 2), 
    #             (2,0), (2,1), (2, 2), 
    # ]

    # actions = [
    #             (0,0), (0,1), (0, 2), 
    #             (1,0), (0,1), (0, 2)
    # ]

    actions = [
                (1,0), (1,1), (1, 2)
    ]

    moved = False

    action = network.choose_action(state)

    if actions[action][1] == 1:
        player_car.rotate(left=True)
    if actions[action][1] == 2:
        player_car.rotate(right=True)
    if actions[action][0] == 1:
        moved = True
        player_car.move_forward()
    if actions[action][0] == 2:
        moved = True
        player_car.move_backward()

    if not moved:
        player_car.reduce_speed()

    return action

def get_beam(surface, angle, pos):
    # returns the hit position and length of a beam
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

        
    return hit_pos, euclidean_distance(pos, hit_pos)

def draw_beam(surface, angle, pos):
    hit_pos, beam_length = get_beam(surface, angle, pos)

    pygame.draw.line(surface, BLUE, pos, hit_pos)
    pygame.draw.circle(surface, GREEN, hit_pos, 3)

    pygame.display.update()


run = True
clock = pygame.time.Clock()
images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
          (FINISH, FINISH_POSITION), (mask_surface, (0, 0))]
player_car = PlayerCar(8, 8)

network = Network(
    state_size = 8,
    action_size = 3,
    learning_rate = 0.2,
    gamma = 0.6,
    epsilon = 0.4
)

episodes = 10000

for episode in range(episodes):
    time_step = 0
    distance_traveled = 0
    last_car_center = player_car.x + RED_CAR.get_width() // 2, player_car.y + RED_CAR.get_height() // 2
    prev_state = [0, 0, 0, 0, 0, 0, 0, 0]
    action = 0

    while run:
        clock.tick(FPS)

        draw(WIN, images, player_car)

        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(f'Episode:{episode}/{episodes}', True, GREEN, BLUE)
        textRect = text.get_rect()
        textRect.center = (130, HEIGHT - 70)
        WIN.blit(text, textRect)

        text = font.render(f'Epsilon:{network.epsilon:.3f}', True, GREEN, BLUE)
        textRect = text.get_rect()
        textRect.center = (130, HEIGHT - 30)
        WIN.blit(text, textRect)

        car_center = player_car.x + RED_CAR.get_width() // 2, player_car.y + RED_CAR.get_height() // 2
        beam_start_angle = -(player_car.angle + 180)

        beam_lengths = []

        for angle in range(beam_start_angle, beam_start_angle + 181, 30):
            draw_beam(WIN, angle, car_center)
            beam_lengths.append(get_beam(WIN, angle, car_center)[1] / 500)

        parallel = beam_lengths[0] + beam_lengths[-1]
        center = abs(beam_lengths[0] - beam_lengths[-1])

        distance_traveled += euclidean_distance(last_car_center, car_center)
        last_car_center = car_center

        state = beam_lengths + [player_car.vel / player_car.max_vel]
        reward = [0, 0, distance_traveled, parallel, center]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        if player_car.collide(mask) != None:
            print('collide')
            reward[0] = 1
            break
            
        finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
        if finish_poi_collide != None:
            if finish_poi_collide[1] == 0:
                print('collide')
                reward[0] = 1
            else:
                print('finish')
                reward[1] = 1
            break

        network.train(prev_state, action, state, reward)

        action = move_cpu(player_car, network, state)

        prev_state = state

        time_step += 1

    if run == False:
        break

    network.update_epsilon()
    player_car.reset()

network.save()

# network = Network(
#     state_size = 6,
#     action_size = 3,
#     learning_rate = 0.001,
#     gamma = 0.9,
#     epsilon = 0.1
# )

# network.load()

# episodes = 10000

# for _ in range(episodes):
#     time_step = 0
#     states = []
#     rewards = []
#     distance_traveled = 0
#     last_car_center = player_car.x + RED_CAR.get_width() // 2, player_car.y + RED_CAR.get_height() // 2

#     while run:
#         clock.tick(FPS)

#         draw(WIN, images, player_car)

#         car_center = player_car.x + RED_CAR.get_width() // 2, player_car.y + RED_CAR.get_height() // 2
#         beam_start_angle = -(player_car.angle + 180)

#         beam_lengths = []

#         for angle in range(beam_start_angle + 50, beam_start_angle + 131, 20):
#             draw_beam(WIN, angle, car_center)
#             beam_lengths.append(get_beam(WIN, angle, car_center)[1] / 500)

#         states.append([beam_lengths, player_car.vel / player_car.max_vel])

#         distance_traveled += euclidean_distance(last_car_center, car_center)
#         last_car_center = car_center

#         rewards.append([0, 0, distance_traveled])

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#                 break

#         move_cpu(player_car, network, states[-1])

#         if player_car.collide(mask) != None:
#             print('collide')
#             rewards[-1][0] = 1
#             break
            
#         finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
#         if finish_poi_collide != None:
#             if finish_poi_collide[1] == 0:
#                 print('collide')
#             else:
#                 print('finish')
#                 rewards[-1][1] = 1
#             break

#         time_step += 1

#     if run == False:
#         break

#     states = []
#     rewards = []
#     player_car.reset()


pygame.quit()
