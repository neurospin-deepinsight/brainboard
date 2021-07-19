# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to display a dynamic board.
"""

# Imports
import sys
import json
from subprocess import Popen, PIPE
import numpy as np
import visdom


class Board(object):
    """ Define a dynamic board: usefull to gather interesting plottings
    during the training.
    """
    def __init__(self, port=8096, host="http://localhost", env="main"):
        """ Init class.

        Parameters
        ----------
        port: int, default 8097
            the port on which the visdom server is launched.
        host: str, default 'http://localhost'
            the host on which visdom is launched.
        env: str, default 'main'
            the environment to be used.
        """
        self.port = port
        self.host = host
        self.env = env
        self.plots = {}
        self.viewer = visdom.Visdom(
            port=self.port, server=self.host, env=self.env)
        self.server = None
        if not self.viewer.check_connection():
            self._create_visdom_server()
        current_data = json.loads(self.viewer.get_window_data())
        for key in current_data:
            self.viewer.close(win=key)

    def __del__(self):
        """ Class destructor.
        """
        if self.server is not None:
            self.server.kill()
            self.server.wait()

    def _create_visdom_server(self):
        """ Starts a new visdom server.
        """
        current_python = sys.executable
        cmd = "{0} -m visdom.server -p {1}".format(current_python, self.port)
        print("Starting visdom server:\n{0}".format(cmd))
        self.server = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def update_image(self, name, images, title=None):
        """ Update image display.

        Parameters
        ----------
        name: str
            the name of the plot to be updated.
        images: array (N,1,X,Y) or (N,3,X,Y)
            the images to be displayed.
        title: str, default None
            the image title.
        """
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(
                "You must define a function that transforms the "
                "predictions into a Nx1xXxY or Nx3xXxY array, with N "
                "the number of images.")
        self.viewer.images(
            images, opts={"title": title or name, "caption": name}, win=name)

    def update_plot(self, name, x, y):
        """ Update plot.

        Parameters
        ----------
        name: str
            the name of the plot to be updated.
        x, y: numbers
            the new point to be displayed.
        """
        self.viewer.line(
            X=np.asarray([x]), Y=np.asarray([y]),
            opts={"title": name, "xlabel": "iterations", "ylabel": name},
            update="append", win=name)

    def update_hist(self, name, labels):
        """ Update histogram.

        Parameters
        ----------
        name: str
            the name of the plot to be updated.
        labels: list of int
            the labels to be displayed.
        """
        self.viewer.bar(
            labels, opts={"title": "name", "caption": "name"}, win=name)
