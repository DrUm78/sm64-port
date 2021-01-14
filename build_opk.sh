#!/bin/sh -e

convert textures/segment2/segment2.05A00.rgba16.png -resize 32x32! build/icon.png
mksquashfs build/us_pc/sm64.us.f3dex2e build/icon.png default."$1".desktop sm64-port-"$1".opk
