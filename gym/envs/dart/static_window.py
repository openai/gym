# Contributors: Wenhao Yu (wyu68@gatech.edu) and Dong Xu (donghsu@gatech.edu)

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene import OpenGLScene
from pydart2.gui.glut.window import *


class StaticGLUTWindow(GLUTWindow):
    def close(self):
        GLUT.glutDestroyWindow(self.window)
        GLUT.glutMainLoopEvent()

    def drawGL(self, ):
        self.scene.render(self.sim)
        GLUT.glutSwapBuffers()

    def runSingleStep(self):
        GLUT.glutPostRedisplay()
        GLUT.glutMainLoopEvent()

    def getGrayscale(self, _width, _height):
        # get compressed grayscale img
        # for end to end learning
        # Do not call it in other case
        # there will be some potential problems
        from PIL import Image
        data = GL.glReadPixels(0, 0,
                               _width, _height,
                               GL.GL_RGBA,
                               GL.GL_UNSIGNED_BYTE)
        img = Image.frombytes("RGBA", (_width, _height), data).convert('L')
        img = np.array(img.getdata(), dtype=np.uint8)
        return img.reshape(_width, _height)

    def getFrame(self):
        self.runSingleStep()
        data = GL.glReadPixels(0, 0,
                               self.window_size[0], self.window_size[1],
                               GL.GL_RGBA,
                               GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8)
        return img.reshape(self.window_size[1], self.window_size[0], 4)[::-1,:,0:3]

    def mykeyboard(self, key, x, y):
        keycode = ord(key)
        key = key.decode('utf-8')
        # print("key = [%s] = [%d]" % (key, ord(key)))

        # n = sim.num_frames()
        if keycode == 27:
            self.close()
            return
        self.keyPressed(key, x, y)

    def run(self, _width=None, _height=None, _show_window=True):
        # Init glut
        self._show_window = _show_window
        GLUT.glutInit(())
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA |
                                 GLUT.GLUT_DOUBLE |
                                 GLUT.GLUT_ALPHA |
                                 GLUT.GLUT_DEPTH)
        if _width is not None and _height is not None:
            GLUT.glutInitWindowSize(_width,_height)
            #self.resizeGL(_width, _height) # this line crashes my program ??
        else:
            GLUT.glutInitWindowSize(*self.window_size)
        GLUT.glutInitWindowPosition(0, 0)
        self.window = GLUT.glutCreateWindow(self.title)
        if not _show_window:
            GLUT.glutHideWindow()

        GLUT.glutDisplayFunc(self.drawGL)
        GLUT.glutReshapeFunc(self.resizeGL)
        GLUT.glutKeyboardFunc(self.mykeyboard)
        GLUT.glutMouseFunc(self.mouseFunc)
        GLUT.glutMotionFunc(self.motionFunc)
        self.initGL(*self.window_size)
