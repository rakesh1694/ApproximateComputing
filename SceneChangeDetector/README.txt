This code is part of VideoChef USENIX ATC 18 paper

The code implements a low overhead histogram difference based scene change detector function.

Steps required:
Need to first initialize scene change detector by calling init_scene_change_detector().
Scene change detector maintains the previous frame histrogram as previous state. Whenever
different_scene() function is called, it calculates the the histogram of given frame and then calculates 
the histogram difference with the previous frame. The function returns True is difference is greater
than 20% threshold.

A sample example describing how to use the function is given inside main() function.