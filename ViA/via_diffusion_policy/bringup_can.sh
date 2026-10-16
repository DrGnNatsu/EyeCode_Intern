#!/bin/bash

# Bring up CAN interfaces with 1 Mbps bitrate
sudo ip link set can_top up type can bitrate 1000000
sudo ip link set can_right up type can bitrate 1000000
sudo ip link set can_left up type can bitrate 1000000
