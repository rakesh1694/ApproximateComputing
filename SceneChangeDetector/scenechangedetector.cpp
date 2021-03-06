#include "stdio.h"
#include <time.h>

#include <stdbool.h>

int prev_hist[256];
int curr_hist[256];

void init_scene_change_detector() {
	for (int i = 0; i < 256; i++) {
		prev_hist[i] = 0;
	}
}

bool different_scene(unsigned char *data, int size) {
	int i = 0;
	for (i = 0; i < 256; i++) {
		curr_hist[i] = 0;
	}
	for (i = 0; i < size; i++) {
		curr_hist[data[i]]++;
	}
	double diff = 0.0;
	for (i = 0; i < 256; i++) {
		if (curr_hist[i] > prev_hist[i]) {
			diff += curr_hist[i] - prev_hist[i];
		}
		else {
			diff += prev_hist[i] - curr_hist[i];
		}
		prev_hist[i] = curr_hist[i];
	}
	diff = diff / (2.0*size);
	return diff > 0.2;
}


/* 
Example explaining how to use scene change detector function
Need to first initialize scene change detector by calling init_scene_change_detector()
Scene change detector maintains the previous frame histrogram as previous state. Whenever
different_scene() function is called, it calculates the the histogram of given frame and calculates 
the histogram difference with the previous frame. The function returns True is difference is greater
than 20% threshold.
*/

#define SIZE_Y 90*160 	//size of Y channel
#define SIZE_U 45*80	//size of U channel
#define SIZE_V 45*80	//size of V channel

int main()
{
	printf("SIZE_Y = %i\n", SIZE_Y);

	FILE *fileptr;
	video_file_path = "../../../../CanaryRawVideo/Video/10.dat"
	fileptr = fopen(video_file_path, "rb");
	unsigned char buffer_y[SIZE_Y];
	unsigned char buffer_u[SIZE_U];
	unsigned char buffer_v[SIZE_V];
	init_scene_change_detector();
	int scene_change_frame[50];
	int count = 0;
	int frame_no = 0;
	printf("unsigned char size -> %i\n", sizeof(unsigned char));
	clock_t start, end;
	double cpu_time_used = 0.0;
	while (1) {
		int done = fread(buffer_y, 1, SIZE_Y, fileptr);
		//printf("frame -> %i, done -> %i\n", frame_no,done);
		if (done == 0) {
			break;
		}
		fread(buffer_u, sizeof(unsigned char), SIZE_U, fileptr);
		fread(buffer_v, sizeof(unsigned char), SIZE_V, fileptr);
		start = clock();
		if (different_scene(buffer_y, SIZE_Y)) {
			scene_change_frame[count] = frame_no;
			count++;
			if(count==50){
				break;
			}
		}
		end = clock();
		cpu_time_used += (end - start);

		frame_no++;
	}
	fclose(fileptr);

	float total_time = (cpu_time_used*1.0) / (CLOCKS_PER_SEC);
	float time_per_frame = total_time / frame_no;
	printf("total time = %f, time per frame = %f\n", total_time, time_per_frame);
	printf("scene changed at following frames: ");
	for (int i = 0; i < count; i++) {
		printf("->%i", scene_change_frame[i]);
	}

    return 0;
}

