# Stage 1: Builder
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

WORKDIR /app

COPY generated-kernel.cu .

# Compile the CUDA code into an object file
RUN nvcc -c generated-kernel.cu -o kernel.o

# Stage 2: Final image
FROM scratch
COPY --from=builder /app/kernel.o /kernel.o
