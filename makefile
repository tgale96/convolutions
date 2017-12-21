# There is certainly a better way to do this.

all:
	nvcc -std=c++11 -o conv conv.cu -lcudnn

debug:
	nvcc -std=c++11 -DDEBUG -o conv conv.cu -lcudnn

backward:
	nvcc -std=c++11 -o backward_conv backward_conv.cu -lcudnn

debug_backward:
	nvcc -std=c++11 -DDEBUG -o backward_conv backward_conv.cu -lcudnn

backward_data:
	nvcc -std=c++11 -o backward_data_conv backward_data_conv.cu -lcudnn

debug_backward_data:
	nvcc -std=c++11 -DDEBUG -o backward_data_conv backward_data_conv.cu -lcudnn

clean:
	rm -f conv backward_conv backward_data_conv
