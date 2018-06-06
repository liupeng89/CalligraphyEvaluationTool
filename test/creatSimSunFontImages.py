# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import

import argparse
import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont

import collections


def load_charset(char_dir):
    with open(char_dir, 'r') as f:
        charset = f.readlines()
        charset = [char.strip() for char in charset]
        return charset

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def draw_example(ch, font, canvas_size, x_offset, y_offset):
    img = draw_single_char(ch, font, canvas_size, x_offset, y_offset)


def font2image(font_dir, charset, char_size, canvas_size, x_offset, y_offset, sample_dir):
    font = ImageFont.truetype(font_dir, size=char_size)

    count = 0
    for char in charset:
        e = draw_single_char(char, font, canvas_size, x_offset, y_offset)
        if e:
            e.save(os.path.join(sample_dir, ("%04d" + char + ".jpg") % count))
            count += 1
            if count % 100 == 0:
                print("process %d chars" % count)




parser = argparse.ArgumentParser(description="Covert SimSun font to images")
parser.add_argument("--char_dir", dest="char_dir", required=True, help="path of the characters")
parser.add_argument("--font_dir", dest="font_dir", required=True, help="path of the source font")

parser.add_argument("--char_size", dest="char_size", type=int, default=512, help="character size")
parser.add_argument("--canvas_size", dest="canvas_size", type=int, default=512, help="canvas size")
parser.add_argument("--x_offset", dest="x_offset", type=int, default=0, help="x offset")
parser.add_argument("--y_offset", dest="y_offset", type=int, default=0, help="y offset")
parser.add_argument("--sample_dir", dest="sample_dir", help="directory to save sample")

args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    charset = load_charset(args.char_dir)

    font2image(args.font_dir, charset, args.char_size, args.canvas_size, args.x_offset, args.y_offset,
               args.sample_dir)
