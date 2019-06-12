---
layout: post
title:  "Understanding memory information from top or free linux commands"
date:   2018-06-12 05:00:00
categories: linux
---

It's hard to have a simple information on how to understand basic `top` and `free` linux commands.

For `top` command:

<img src="{{ site.url }}/img/top.png" >

- on the *Memory* line (corresponding to RAM):

$$ \text{total} = \text{free} + \text{used} + \text{buff/cache} $$

ie

$$ 1166060 + 57391012 + 73387784 = 131944856 \leftarrow 13194485+ $$

where + is hidding one char, and

- on the *Swap* line, the total corresponds to part of the data that has been saved to Disk as well as remaning disk space (Disk partition size) :

$$ \text{total} = \text{free} + \text{used} $$

$$ 2097148 = 2097148 + 0 $$

When running `free -m` command a few second later:

<img src="{{ site.url }}/img/free.png" >

we find almost same values, and exact same value for the total (physical RAM size):

$$ \text{total} = \text{free} + \text{used} + \text{buff/cache} $$

$$ 131944856 = 57390792 + 1166276 + 73387788 $$
