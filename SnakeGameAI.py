import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

#font(değiştireceğim sonra)
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):# bunu da mesela right ve Right farklı olarak algılanmasın diye yazdık
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 20

#renkler(sonra değiştir olmazsa)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 180, 0)
BLACK = (0, 0, 0)

class SnakeGameAI:
    def __init__(self, w = 640, h = 480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)#videoda self.pt idi
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        '''[self.pt, 
                      Point(self.pt.x-BLOCK_SIZE, self.pt.y),
                      Point(self.pt.x-(2*BLOCK_SIZE), self.pt.y)]'''
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            '''if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN'''

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):#eğer food'u yemezse yani ölmeden ama durumunu da değiştirmeden devam ederseyi de kontrol edeceğiz.
            game_over = True
            reward = -10 # buraalrı bir de reward += 10 şeklinde dene
            return reward, game_over, self.score
        
        if self.head == self.food:
            reward = 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()


        self._update_ui()
        self.clock.tick(SPEED)

        game_over = False
        return reward, game_over, self.score
    
    def is_collision(self, pt = None):
        if pt is None:
            pt = self.head 
        if pt.x >= self.w - BLOCK_SIZE or pt.x < 0 or pt.y >= self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0 , 0])
        pygame.display.flip()# bu olmadan değişiklikleri göremezmişiz

    def _move(self, action):
        #straight, left ya da right

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #değişmiyor
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4 #pek anlamadım
            new_dir = clock_wise[next_idx]# sağa dönüyor r->d->l->u
        else:
            next_idx = (idx - 1) % 4 
            new_dir = clock_wise[next_idx]#sola dönüyor r->u->l->d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
    
'''if __name__ == '__main__':
    game = SnakeGameAI()

    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break
    


    pygame.quit()'''
