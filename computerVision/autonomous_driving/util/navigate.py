def navigate(self, center_offset, danger_zone):
    if danger_zone:
        return "BRAKE (Obstacle Ahead!)", (0, 0, 255)
    elif center_offset == -1:
        return "STEER RIGHT (Adjust Lane)", (0, 255, 255)
    elif center_offset == 1:
        return "STEER LEFT (Adjust Lane)", (0, 255, 255)
    else:
        return "CRUISE (Path Clear)", (0, 255, 0)
