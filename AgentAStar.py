import pygame
import random
import numpy as np
import heapq
from SnakeGameAI import SnakeGameAI, Direction, Point, BLOCK_SIZE

class AgentAStar:
    #A* algoritması ile start noktasından goal noktasına en kısa yolu hesaplar.
    def astar(self, start, goal, obstacles, grid_width, grid_height):

        #heuristic, manhattan mesafesi(iki nokta arasındaki en kısa mesafe)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = []#henüz işlenmemiş ve değerlendirilmesi gereken nodeların bulunduğu liste
        heapq.heappush(open_set, (0, start))#(pq -min heap)
        came_from = {}#hangi nokta nerden geldi
        g_score = {start: 0}#başlangıç noktasına olan maliyet
        f_score = {start: heuristic(start, goal)}#g_score + heuristic
        closed_set = set()
        
        #burada hedefe ulaşılırsa path oluşturuyoruz
        while open_set:
            current_f, current = heapq.heappop(open_set)# min f_score olan nokta
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]#yolu geriye doğru takip ettik
                    path.append(current)
                path.reverse()#yolu baştan sona sıraladık
                return path
            
            closed_set.add(current)

            #4 farklı komşuya bakıyor; sağ, sol, yukarı, aşağı
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor[0] < 0 or neighbor[0] >= grid_width or neighbor[1] < 0 or neighbor[1] >= grid_height: # sınırların içinde mi?
                    continue
                if neighbor in obstacles:# engel varsa(yılan gövdesi)
                    continue
                if neighbor in closed_set: #zaten işlendi mi?
                    continue
                

                #daha iyi yol bulunursa bilgiler güncellenip queueya ekleniyor
                tentative_g = g_score[current] + 1 #yeni maliyet
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    #A* ile yeme ulaşmak için yapılacak hamleleri belirliyoruz
    def get_action(self, game: SnakeGameAI):
        # ızgara boyutları
        grid_width = game.w // BLOCK_SIZE
        grid_height = game.h // BLOCK_SIZE

        # yılanın kafası ve yeme ızgara koordinatlarına çevriliyor
        start = (int(game.head.x // BLOCK_SIZE), int(game.head.y // BLOCK_SIZE))
        goal = (int(game.food.x // BLOCK_SIZE), int(game.food.y // BLOCK_SIZE))
        
        # engel konumları; yılanın gövdesi (kafa hariç)
        obstacles = set()
        for pt in game.snake[1:]:
            obstacles.add((int(pt.x // BLOCK_SIZE), int(pt.y // BLOCK_SIZE)))
        
        path = self.astar(start, goal, obstacles, grid_width, grid_height)#a* ile yol bul
        
        if path is None or len(path) < 2:
            # Yol bulunamazsa düz git
            return [1, 0, 0]
        
        #hesaplanan yolun ilk adımı (başlangıç noktasının ardından gelen)
        next_cell = path[1]
        head_grid = start # yılanın şu anki kafasının bulunduğu cell
        
        # hedef hücreye göre istenen yön belirleniyor
        desired_direction = None
        if next_cell[0] > head_grid[0]:#x koordinatından büyükse sağa git
            desired_direction = Direction.RIGHT 
        elif next_cell[0] < head_grid[0]:#x koordinatından küçükse sola git
            desired_direction = Direction.LEFT
        elif next_cell[1] > head_grid[1]:#y koordinatından büyükse aşağı git
            desired_direction = Direction.DOWN
        elif next_cell[1] < head_grid[1]:#y koordinatından küçükse yukarı git
            desired_direction = Direction.UP
        
        # yılanın hareket edebileceği yönleri saat yönünde sıralıyor
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = clock_wise.index(game.direction)
        
        if desired_direction == game.direction:#mevcut yön ile istenen yön aynı ise
            return [1, 0, 0]  #düz git
        desired_idx = clock_wise.index(desired_direction)
        if (current_idx + 1) % 4 == desired_idx:#hedef yön mevcut yönün saat yönünde  bir sonraki yönü ise 
            return [0, 1, 0]#sağa dön
        elif (current_idx - 1) % 4 == desired_idx:#hedef yön mevcut yönün saat yönünün tersine bir önceki yönü ise
            return [0, 0, 1]#sola dön
        else:
            return [1, 0, 0]#ters yönde ise (genel olarak istemiyoruz) düz git

if __name__ == '__main__':
    game = SnakeGameAI()
    agent = AgentAStar()
    while True:
        action = agent.get_action(game)
        reward, game_over, score = game.play_step(action)
        if game_over:
            print("Oyun bitti! Skor:", score)
            game.reset()
