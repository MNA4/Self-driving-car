import pygame, math, numpy
from pygame.locals import *
pygame.init()

class NeuralNetwork:
    def __init__(self, n_input, n_output, n_hlayers, l_hlayers):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hlayers = n_hlayers
        self.l_hlayers = l_hlayers
#        self.inodes = []
        self.hnodes = []
        self.onodes = []
        self.n_path = n_hlayers * n_input + n_output * n_hlayers + n_hlayers**2 * (l_hlayers - 1)
        self.n_node = n_input + n_output + n_hlayers * l_hlayers

#        for i in range(n_input):
#            self.inodes.append(Node())
        for i in range(l_hlayers):
            self.hnodes.append([])
            for j in range(n_hlayers):
                self.hnodes[i].append(0)
        for i in range(n_output):
            self.onodes.append(0)
    def calculate(self, biases, weights, inputs):
        w_index = 0
        b_index = 0
        for i in range(self.l_hlayers):
            for j in range(self.n_hlayers):
                if i == 0:
                    self.hnodes[i][j] = activation(biases[b_index] + numpy.sum(weights[w_index:w_index+len(inputs)]*inputs))
                    w_index += len(inputs)
                else:
                    self.hnodes[i][j] = activation(biases[b_index] + numpy.sum(weights[w_index:w_index+self.n_hlayers]*self.hnodes[i-1]))
                    w_index += self.n_hlayers
                b_index += 1
        for i in range(self.n_output):
            self.onodes[i] = activation(biases[b_index] + numpy.sum(weights[w_index:w_index+self.n_hlayers]*self.hnodes[-1]))
            w_index += self.n_hlayers
            b_index += 1
        return self.onodes
    
class Car:
    def __init__(self, pos, angle):
        self.img = pygame.transform.scale(pygame.image.load('car.png'), (18, 38))
        self.map = surf
        self.pos = pos
        self.angle = angle
        self.neural_network = NeuralNetwork(5, 1, 3, 1)
        self.dead = False
        
    def turn(self, angle):
        self.forward(-9)
        self.angle -= angle
        self.forward(9)
        
    def forward(self, speed = 1):
        s = math.sin(math.radians(self.angle))* speed
        c = math.cos(math.radians(self.angle))* speed
        self.pos[0] -= s
        self.pos[1] -= c

    def render(self, screen):
        rotated_car = pygame.transform.rotozoom(self.img, self.angle, 1)
        screen.blit(rotated_car, [self.pos[0]-rotated_car.get_width()/2,
                                  self.pos[1]-rotated_car.get_height()/2,])
    def is_colliding(self):
        rotated_car = pygame.transform.rotozoom(self.img, self.angle, 1)
        car_mask = pygame.mask.from_surface(rotated_car)
        return mask.overlap(car_mask, [self.pos[0]-rotated_car.get_width()/2,
                                       self.pos[1]-rotated_car.get_height()/2,])

# functions
def activation(gamma): # Thanks to: https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def normalize(x, y):
    d = math.sqrt(x ** 2 + y ** 2)
    if d == 0:
        return 0, 1
    
    return x / d, y / d

def rotate(v1, v2, normalized = False):
    if normalized:
        nvx, nvy = v1
    else:
        nvx, nvy = normalize(v1[0], v1[1])
        
    return nvy*v2[0]+nvx*v2[1], nvy*v2[1]-nvx*v2[0]

def cast_ray(o, d):
    p = list(o)
    d = normalize(*d)
    for i in range(200):
        if 0<p[0]<WIDTH and 0<p[1]<HEIGHT and mask.get_at(p):
            pygame.draw.line(screen, (0,0,255), o, p)
            pygame.draw.circle(screen, (0,0,0), p, 2)
            return i/200
        p[0]+=d[0]
        p[1]+=d[1]
    pygame.draw.line(screen, (0,0,255), o, p)
    return 200

WIDTH = 512
HEIGHT = 512
FPS = 1000000


surf = pygame.image.load('map.png')
mask = pygame.mask.from_threshold(surf, (181, 230, 29), (1,1,1))

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 22)
mutation_rate = 1/100
b_biases = 0
b_weights = 0
death_count = 0
t = 0
personal_best = 0
cars_count = 10
cars = []
for i in range(cars_count):
    c = Car([43, 43],-90)
    cars.append(c)
    c.biases = b_biases + (numpy.random.rand(c.neural_network.n_node)*2-1)*mutation_rate
    c.weights = weights = b_weights + (numpy.random.rand(c.neural_network.n_path)*2-1)*mutation_rate
    
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
    screen.fill((0,0,0))
    screen.blit(surf, (0,0))
    for i in cars:
        if not i.dead:
            i.forward(1)
            s = math.sin(math.radians(i.angle))
            c = math.cos(math.radians(i.angle))
            rotated_car = pygame.transform.rotozoom(i.img, i.angle, 1)
            pos = i.pos
            d1 = cast_ray(pos, rotate((s,c), (-.25,-1)))
            d2 = cast_ray(pos, rotate((s,c), (0,-1)))
            d3 = cast_ray(pos, rotate((s,c), (.25,-1)))
            d4 = cast_ray(pos, rotate((s,c), (-.5,-1)))
            d5 = cast_ray(pos, rotate((s,c), (.5,-1)))
            i.turn(i.neural_network.calculate(i.biases, i.weights, [d1,d2,d3,d4,d5])[0]*180-90)
            i.render(screen)
            if i.is_colliding():
                if death_count >= cars_count-1:                    
                    personal_best = t
                    death_count = 0
                    b_biases = i.biases
                    b_weights = i.weights              
                    t = 0
                    for j in cars:
                        j.dead = False
                        j.pos = [43, 43]
                        j.angle = -90
                        j.biases = b_biases + (numpy.random.rand(j.neural_network.n_node)*2-1)*mutation_rate
                        j.weights = weights = b_weights + (numpy.random.rand(j.neural_network.n_path)*2-1)*mutation_rate
                else:
                    i.dead = True
                    death_count += 1
    t+=1
    screen.blit(font.render('Personal best: %s frames'%personal_best, True, (0,0,0)), (10,10))
    pygame.display.flip()
    clock.tick(FPS)
pygame.quit()

