import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pyglet
from app import SoulStreamApp


def main():
    app = SoulStreamApp()
    pyglet.clock.schedule_interval(lambda dt: None, 1 / 60)
    pyglet.app.run()


if __name__ == "__main__":
    main()
