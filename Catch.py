import pygame as pg
import numpy as np
import theano
import theano.tensor as T
import pandas as pd
from collections import Counter
import random
import sys
from Constants_Catch import *
from math import sqrt

#Game
class CatchGame():
    def __init__(self, com_pos = None, play_pos = None):
        pg.init()
        if not com_pos:
            com_pos = []
        if not play_pos:
            play_pos = []
        if pg.font:
            self.font = pg.font.Font(None,30)
        else:
            self.font = None
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(SCREEN_SIZE)
        self.start_message_time = 200
        self.start_time = 0
        self.caption_text = "AI: A Game of Catch"
        self.init_game()

    def eventLoop(self):
        'process events (specifically closing the game window)'
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

    def init_game(self):
        ###########Initialize Sprites
        #sprite_sheet = SpriteSheet("tiles_spritesheet.png")
        #banan_sheet = SpriteSheet("Banan.png")
        #self.paddle_sprt = sprite_sheet.get_image(648, 648, 70, 40)
        #self.ball_sprt = banan_sheet.get_image(0,0,32,32)
        #self.fireball_sprt = banan_sheet.get_image(0,32,32,32)
        #self.fireball_aftersprt = banan_sheet.get_image(0,64,32,32)

        self.fps = FPS
        self.score = 0
        self.state = STATE_PLAY
        self.start_message = "Start!"
        pg.display.set_caption(self.caption_text)
        self.camera = Camera(simple_camera_func, GAME_SIZE[0], GAME_SIZE[1])
        self.lshift = 0
        self.lctrl = 0

        self.banner   = pg.Rect(0, SCREEN_SIZE[1]/2 - BANNER_HEIGHT/2, SCREEN_SIZE[0], BANNER_HEIGHT)
        self.init_objects()

    def init_objects(self):
        self.create_enter()
        self.create_exit()
        self.create_terrain()
        self.create_players()
        self.is_possible()
        #self.singulars = [self.enter_E, self.exit_E, self.player]
        #self.entities = self.grounds + self.walls + self.singulars
    """def create_ground(self):
        self.xloc = 0
        while self.xloc < GAME_SIZE[0]:
            self.yloc = 0
            while self.yloc < GAME_SIZE[1]:
                self.grounds.append(Ground(self.xloc,self.yloc))
                self.yloc += OBJ_H
            self.xloc += OBJ_W
        self.grounds = tuple(self.grounds)"""

    def create_terrain(self):
        self.grounds = []
        self.walls = []
        self.xloc = 0
        while self.xloc < GAME_SIZE[0]:
            self.yloc = 0
            while self.yloc < GAME_SIZE[1]:
                topleft = (self.xloc,self.yloc)
                if topleft != self.exit_E.shape.topleft and topleft != self.enter_E.shape.topleft:
                    if random.random() < WALL_CHANCE:
                        self.walls.append(Wall(self.xloc,self.yloc))
                    else:
                        self.grounds.append(Ground(self.xloc,self.yloc))
                self.yloc += OBJ_H
            self.xloc += OBJ_W
        self.grounds = tuple(self.grounds)
        self.walls = tuple(self.walls)

    def create_enter(self):
        xpos = random.random()*GAME_SIZE[0]
        ypos = random.random()*GAME_SIZE[1]
        xpos -= xpos % OBJ_W
        ypos -= ypos % OBJ_H
        self.enter_E = Elevator(xpos, ypos)

    def create_players(self):
        self.player = Player(self.enter_E.shape.left, self.enter_E.shape.top)
        self.computer = Computer(self.enter_E.shape.left, self.enter_E.shape.top)
        self.moving_entities = [self.player, self.computer]

    def create_exit(self):
        while True:
            xpos = random.random()*GAME_SIZE[0]
            ypos = random.random()*GAME_SIZE[1]
            xpos -= xpos % OBJ_W
            ypos -= ypos % OBJ_H
            if xpos == self.enter_E.shape.left and ypos == self.enter_E.shape.top:
                print('same location chosen, reseting exit')
                continue
            break
        self.exit_E = Elevator(xpos, ypos, end = True)

    def is_possible(self):
        end = self.exit_E.shape.center
        interpoints = [self.enter_E.shape.center]
        i=0
        wallcenters = tuple([wall.shape.center for wall in self.walls])
        while True:
            i+=1
            New = 0
            EndTest = 0
            checked = interpoints.copy()
            for point in checked:
                newpoints = ((point[0] + OBJ_W, point[1]),(point[0] - OBJ_W, point[1]),(point[0], point[1]+OBJ_H),(point[0], point[1]-OBJ_H))
                for newpoint in newpoints:
                    if 0<newpoint[0]<GAME_SIZE[0] and 0<newpoint[1]<GAME_SIZE[1]:
                        inwall = 0
                        if newpoint in interpoints:
                            continue
                        if newpoint in wallcenters:
                            inwall = 1
                        if not inwall:
                            if newpoint == end:
                                print('end found from enter')
                                return True
                            interpoints.append(newpoint)
                            New += 1
                            if New > NEWPOINT_REQUIREMENT:
                                print('enough new points that there is probably a path')
                                EndTest = 1
                                break
                        if EndTest:
                            break
                        continue
                if EndTest:
                    break
            if EndTest:
                break
            print(i)
            if New == 0:
                print('no path to end, resetting')
                self.state = STATE_RESET
                break
        if EndTest:
            end = self.enter_E.shape.center
            interpoints = [self.exit_E.shape.center]
            i=0
            while True:
                i+=1
                New = 0
                checked = interpoints.copy()
                for point in checked:
                    newpoints = ((point[0] + OBJ_W, point[1]),(point[0] - OBJ_W, point[1]),(point[0], point[1]+OBJ_H),(point[0], point[1]-OBJ_H))
                    for newpoint in newpoints:
                        if 0<newpoint[0]<GAME_SIZE[0] and 0<newpoint[1]<GAME_SIZE[1]:
                            inwall = 0
                            if newpoint in interpoints:
                                continue
                            if newpoint in wallcenters:
                                inwall = 1
                            if not inwall:
                                if newpoint == end:
                                    print('end found from enter')
                                    return True
                                interpoints.append(newpoint)
                                New += 1
                                if New > NEWPOINT_REQUIREMENT:
                                    print('enough new points that there is probably a path')
                                    return True
                            continue
                print(i)
                if New == 0:
                    print('no path to end, resetting')
                    self.state = STATE_RESET
                    break

    def draw_ground(self):
        for obj_ground in self.grounds:
            dist = distance(obj_ground.shape.center, self.player.shape.center)

            #intersect_point([obj_ground.shape.center, self.player.shape.center], [])
            if dist < self.player.max_vision:
                comdist = distance(obj_ground.shape.center, self.computer.shape.center)
                #obj_ground.found = True
                tile = self.camera.apply(obj_ground)
                lighting = max((1-dist/self.player.vision), ((1-comdist/self.computer.vision) * min(1,1-(dist/self.player.max_vision*FADEOUT_CONSTANT-(FADEOUT_CONSTANT-1)))))
                pg.draw.rect(self.screen, [max(i*lighting,0) for i in C_GROUND], tile)
            elif obj_ground.found:
                tile = self.camera.apply(obj_ground)
                pg.draw.rect(self.screen, C_GROUND_FOUND, tile)

    def walls_seen(self):
        self.walls_in_vision = []
        for obj_wall in self.walls:
            dist = distance(obj_wall.shape.center, self.player.shape.center)
            comdist = distance(obj_wall.shape.center, self.computer.shape.center)
            if dist < self.player.max_vision:
                lighting = max((1-dist/self.player.vision), ((1-comdist/self.computer.vision) * min(1,1-(dist/self.player.max_vision*FADEOUT_CONSTANT-(FADEOUT_CONSTANT-1)))))
                self.walls_in_vision.append((obj_wall, dist, lighting, seen_edges(self.player.shape, obj_wall.shape)))

    def draw_walls(self): #Might be able to merge walls_seen into this function as long as walls are drawn first.
        for wall in self.walls_in_vision:
            obj_wall = wall[0]
            lighting = wall[2]
            #obj_wall.found = True
            #self.walls_seen.append(tile)
            tile = self.camera.apply(obj_wall)
            pg.draw.rect(self.screen, [i*lighting for i in C_WALL], tile)
            #elif obj_wall.found:
            #    tile = self.camera.apply(obj_wall)
            #    pg.draw.rect(self.screen, C_WALL_FOUND, tile)

    def draw_enter(self):
        dist = distance(self.enter_E.shape.center, self.player.shape.center)
        comdist = distance(self.enter_E.shape.center, self.computer.shape.center)
        if dist < self.player.max_vision:
            #self.enter_E.found = True
            enter_tile = self.camera.apply(self.enter_E)
            lighting = max((1-dist/self.player.vision), ((1-comdist/self.computer.vision) * min(1,1-(dist/self.player.max_vision*FADEOUT_CONSTANT-(FADEOUT_CONSTANT-1)))))
            pg.draw.rect(self.screen, [max(0,i*lighting) for i in C_ENTER], enter_tile)
        elif self.enter_E.found:
            enter_tile = self.camera.apply(self.enter_E)
            pg.draw.rect(self.screen, C_ENTER_FOUND, enter_tile)
    def draw_exit(self):
        dist = distance(self.exit_E.shape.center, self.player.shape.center)
        comdist = distance(self.exit_E.shape.center, self.computer.shape.center)
        if dist < self.player.max_vision:
            #self.exit_E.found = True
            exit_tile = self.camera.apply(self.exit_E)
            lighting = max((1-dist/self.player.vision), ((1-comdist/self.computer.vision) * min(1,1-(dist/self.player.max_vision*FADEOUT_CONSTANT-(FADEOUT_CONSTANT-1)))))
            pg.draw.rect(self.screen, [max(0,i*lighting) for i in C_EXIT], exit_tile)
        elif self.exit_E.found:
            exit_tile = self.camera.apply(self.exit_E)
            pg.draw.rect(self.screen, C_EXIT_FOUND, exit_tile)

    def draw_shadows(self):
        polyshapes = []
        for info in self.walls_in_vision:
            #wall = info[0]
            dist = info[1]
            edges = info[3]
            diff_edges = ((self.player.shape.center[0]-edges[0][0],self.player.shape.center[1]-edges[0][1]), (self.player.shape.center[0]-edges[1][0],self.player.shape.center[1]-edges[1][1])) #Put together the differences into a tuple of tuples
            polyshapes.append((edges[0],[edges[0][x]-diff_edges[0][x]*PLAYER_MAX_VISION*3/dist for x in range(2)],[edges[1][x]-diff_edges[1][x]*PLAYER_MAX_VISION*3/dist for x in range(2)],edges[1])) #Make sure edges go far enough to do cover entire lit area
        for j in range(len(self.walls_in_vision)):
            rect = self.walls_in_vision[j][0].shape
            i=0
            draw = True
            for polyshape in polyshapes:
                if i == j:
                    continue
                shadow_src = self.walls_in_vision[i][0]
                if Rect_in_Shadow(shadow_src, polyshape, rect, self.player):
                    draw = False
                    break
                i+=1
            if draw:
                shadow = self.camera.apply(polyshape)
                pg.draw.polygon(self.screen, BLACK, shadow)



    def draw_players(self):
        play_tile = self.camera.apply(self.player)
        pg.draw.rect(self.screen, C_PLAYER, play_tile)

        #Computer Player
        dist = distance(self.computer.shape.center, self.player.shape.center)
        if dist < self.player.max_vision:
            com_tile = self.camera.apply(self.computer)
            lighting = min(1,1-(dist/self.player.max_vision*FADEOUT_CONSTANT-(FADEOUT_CONSTANT-1)))
            pg.draw.rect(self.screen, [i * lighting for i in C_COMPUTER], com_tile)


    def check_input(self):
        for e in self.moving_entities:
            e.centerxprevious = e.shape.centerx
            e.centeryprevious = e.shape.centery
        keys = pg.key.get_pressed()
        if keys[pg.K_BACKSPACE]:
            self.shade = False
        else:
            self.shade = True
        if keys[pg.K_LSHIFT]:
            self.player.vision = PLAYER_VISION_LIMITED
            self.player.max_vision = PLAYER_MAX_VISION
        elif keys[pg.K_LCTRL]:
            self.player.vision = PLAYER_VISION_SILENCE
            self.player.max_vision = PLAYER_MAX_VISION
        elif keys[pg.K_BACKSLASH]:
            self.player.vision = PLAYER_VISION_SUPER
            self.player.max_vision = PLAYER_MAX_VISION_SUPER
            self.shade = False
        else:
            self.player.vision = PLAYER_VISION
            self.player.max_vision = PLAYER_MAX_VISION
        self.player.move_step = self.player.movement_func(keys)
        self.player.shape.topleft = [self.player.shape.topleft[x] + self.player.move_step[x] * self.player.walkspeed for x in range(2)]

        if self.start_time % self.fps == 0:
            self.computer.move_step = self.computer.movement_func()
        self.computer.shape.topleft = [self.computer.shape.topleft[x] + self.computer.move_step[x] * self.computer.walkspeed for x in range(2)]
        if keys[pg.K_SPACE] and self.player.shape.colliderect(self.exit_E.shape):
            print("you won!")
            self.state = STATE_WON
        pass

    def handle_collisions(self):
        for obj_mover in self.moving_entities:
            for wall in self.walls:
                obj_mover.shape.left = max(0, min(GAME_SIZE[0]-obj_mover.shape.width, obj_mover.shape.left))
                obj_mover.shape.top = max(0, min(GAME_SIZE[1]-obj_mover.shape.height, obj_mover.shape.top))
                if obj_mover.shape.colliderect(wall.shape):
                    xdiff = abs(wall.shape.centerx - obj_mover.centerxprevious)
                    ydiff = abs(wall.shape.centery - obj_mover.centeryprevious)
                    if xdiff == ydiff:
                        xdiff += round(random.random())*2-1
                    if xdiff > ydiff:
                        if obj_mover.move_step[0] > 0:
                            obj_mover.shape.right = wall.shape.left-1
                        else:
                            obj_mover.shape.left = wall.shape.right+1
                    elif xdiff < ydiff:
                        if obj_mover.move_step[1] > 0:
                            obj_mover.shape.bottom = wall.shape.top-1
                        else:
                            obj_mover.shape.top = wall.shape.bottom+1
        pass


    def Update(self):
        self.check_input()
        self.handle_collisions()
        self.camera.update(self.player)
        self.walls_seen()
        pass

    def Draw(self):
        self.screen.fill(BLACK) #fill background with black
        self.draw_ground()
        self.draw_walls()
        self.draw_enter()
        self.draw_exit()
        self.draw_players()
        if self.shade:
            self.draw_shadows()
        pass

    def show_message(self,message): #Show message on screen
        if self.font:
            size = self.font.size(message) #use message font size
            font_surface = self.font.render(message,False, TEXT) #make font surface
            # Put font in center of screen
            x = (SCREEN_SIZE[0] - size[0]) / 2
            y = (SCREEN_SIZE[1] - size[1]) / 2
            pg.draw.rect(self.screen, BLACK, self.banner) #place banner on screen
            self.screen.blit(font_surface, (x,y)) #Place on screen

    def Loop(self):
        self.eventLoop()
        self.clock.tick(self.fps) #set fps to 50
        self.Update()
        self.Draw()
        if self.state == STATE_PLAY:
            if self.start_time < self.start_message_time:
                pass
                #self.show_message(self.start_message)
        pg.display.flip()
        self.start_time+=1


#Objects in Game
class Ground():
    def __init__(self, x, y, w=OBJ_W, h=OBJ_H):
        self.x = x
        self.y = y
        self.shape   = pg.Rect(x,y,w,h)
        self.distance = 0
        self.found = False

class Wall():
    def __init__(self, x, y, w=OBJ_W, h=OBJ_H):
        self.shape   = pg.Rect(x,y,w,h)
        self.distance = 0
        self.found = False

class Elevator():
    def __init__(self, x, y, w=OBJ_W, h=OBJ_H, end = False):
        self.shape = pg.Rect(x,y,w,h)
        self.end = end
        self.distance = 0
        self.found = False

#Camera (for sidescrolling)
class Camera(object):
    def __init__(self, camera_func, width, height):
        self.camera_func = camera_func
        self.state = pg.Rect(0, 0, width, height)

    def apply(self, target):
        if type(target) == list or type(target) == tuple:
            return [[i[j] + self.state.topleft[j] for j in range(2)] for i in target]
        return target.shape.move(self.state.topleft)

    def update(self, target):
        self.state = self.camera_func(self.state, target.shape)

class Player():
    def __init__(self, x, y, w=PLAYER_WIDTH, h=PLAYER_HEIGHT):
        self.xprevious = x+w/2
        self.yprevious = y+h/2
        self.w = w
        self.h = h
        self.shape = pg.Rect(x,y,w,h)
        self.walkspeed = PLAYER_WALKSPEED
        self.vision = PLAYER_VISION
        self.max_vision = PLAYER_MAX_VISION
        self.move_step = (0,0)
    def movement_func(self, keys):
        return (keys[pg.K_RIGHT]-keys[pg.K_LEFT],keys[pg.K_DOWN]-keys[pg.K_UP])

class Computer(Player):
    def movement_func(self):
        return (random.random()*2-1,random.random()*2-1)

def simple_camera_func(camera, target_rect):
    l, t, _, _ = target_rect # l = left,  t = top
    _, _, w, h = camera      # w = width, h = height
    return pg.Rect(-l+SCREEN_SIZE[0]/2, -t+SCREEN_SIZE[1]/2, w, h)

def complex_camera_func(camera, target_rect):
    l, t, _, _ = target_rect
    _, _, w, h = camera
    l, t, _, _ = -l+SCREEN_SIZE[0]/2, -t+SCREEN_SIZE[1]/2, w, h

    l = min(0, l)                           # stop scrolling at the left edge
    l = max(-(camera.width-SCREEN_SIZE[0]), l)   # stop scrolling at the right edge
    t = max(-(camera.height-SCREEN_SIZE[1]), t) # stop scrolling at the bottom
    t = min(0, t)                           # stop scrolling at the top
    return pg.Rect(l, t, w, h)

def distance(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def Rect_in_Shadow(wall, shadow, rect, player):
    pr1 = shadow[0]
    pr2 = shadow[3]
    diff1 = [player.shape.center[i]-wall.shape.center[i] for i in range(2)]
    pr1_1 = shadow[1]
    pr2_1 = shadow[2]
    diff2 = [player.shape.center[i]-rect.center[i] for i in range(2)]
    if diff1[0]*diff2[0] < 0 and diff1[1]*diff2[1]:
        return False

    orientation = [0,0]
    if pr1[0] > pr2[0]:
        orientation[0] = 1
    else:
        orientation[0] = -1
    if pr1[1] > pr2[1]:
        orientation[1] = 1
    else:
        orientation[1] = -1
    if orientation[0] == orientation[1]:
        r1 = rect.topleft
        r2 = rect.bottomright
    else:
        r1 = rect.topright
        r2 = rect.bottomleft

    if orientation[0]:
        if r1[1] < x_line(pr1,pr1_1,r1[0]) and r2[1] > x_line(pr2, pr2_1, r2[0]):
            return True


def x_line(p1,p2,x1):
    """
    Returns the y position on a line made by two points at given x position
    """
    if p1[0] < p2[0]:
        minx = p1
        maxx = p2
    else:
        minx = p2
        maxx = p1
    xdiff = maxx[0] - minx[0]
    ydiff = maxx[1] - minx[1]
    rise = ydiff / xdiff
    offset = minx[0]
    return rise*(x1-offset) + offset

def seen_edges(rectWatch, rectSeen):
    """
    Returns the seen edges of a rectangular objects.
    """
    [xdiff,ydiff] = [rectWatch.center[x]-rectSeen.center[x] for x in range(2)]
    if -rectSeen.w/2 < xdiff < rectSeen.w/2:
        if ydiff>0:
            return (rectSeen.bottomright, rectSeen.bottomleft)
        else:
            return (rectSeen.topright, rectSeen.topleft)
    elif -rectSeen.h/2 < ydiff < rectSeen.h/2:
        if xdiff>0:
            return (rectSeen.topright, rectSeen.bottomright)
        else:
            return (rectSeen.topleft, rectSeen.bottomleft)
    else:
        if xdiff * ydiff > 0:
            return (rectSeen.topright, rectSeen.bottomleft)
        else:
            return (rectSeen.topleft, rectSeen.bottomright)
    """if xdiff > rectSeen.w/2:
        x = rectSeen.right
    elif xdiff < -rectSeen.w/2:
        x = rectSeen.left
    else:
        x=None
    if ydiff > rectSeen.h/2:
        y = rectSeen.bottom
    elif ydiff < -rectSeen.h/2:
        y = rectSeen.top
    else:
        y=None"""

#def intersect_point(l1, l2): #takes lines in forms of points, gets intersectionpoint


if __name__ == '__main__':
    for i in range(20):
        game = CatchGame()
        while True:
            game.Loop()
            if game.state == STATE_WON or game.state == STATE_RESET:
                break
    print('thanks for playing!  Run the command "python3 Catch.py" if you want to play again.')
