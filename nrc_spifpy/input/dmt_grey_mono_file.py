#!/usr/bin/env python
# coding: utf-8

from . import DMTGreyFile


class DMTGreyMonoFile(DMTGreyFile):

    def extract_images(self, frame, frame_decomp, date):
        images = super().extract_images(frame, frame_decomp, date, mono=True)

        return images