// Main entry point for kernel configuration system
export * from './types.js';
export * from './utils.js';
export * from './registry.js';
export * from './strategies/index.js';

import { KernelConfigRegistry } from './registry.js';
import {
  EmbeddingKernelStrategy,
  PositionalEncodingStrategy,
  DenseForward2DStrategy,
  DenseForward3DStrategy,
  SplitHeadsStrategy,
  ConcatHeadsStrategy,
  MatMulTiledStrategy,
  SoftmaxStrategy,
  FusedScaleSoftmaxStrategy,
  LayerNormStrategy,
  ElementwiseStrategy,
} from './strategies/index.js';

/**
 * Creates and initializes a kernel configuration registry with all known strategies
 */
export function createKernelConfigRegistry(): KernelConfigRegistry {
  const registry = new KernelConfigRegistry();
  
  // Register embedding strategies
  registry.register('embedding_forward', new EmbeddingKernelStrategy());
  registry.register('positional_encoding_forward', new PositionalEncodingStrategy());
  
  // Register dense layer strategies
  registry.register('dense_forward_2d', new DenseForward2DStrategy());
  registry.register('cublas_dense_forward_2d', new DenseForward2DStrategy());
  registry.register('dense_forward_3d', new DenseForward3DStrategy());
  registry.register('cublas_dense_forward_3d', new DenseForward3DStrategy());
  
  // Register attention strategies
  registry.register('split_heads_forward', new SplitHeadsStrategy());
  registry.register('concat_heads_forward', new ConcatHeadsStrategy());
  registry.register('batched_matmul_transpose_b_tiled', new MatMulTiledStrategy());
  registry.register('batched_matmul_tiled', new MatMulTiledStrategy());
  
  // Register activation/normalization strategies
  registry.register('softmax_forward', new SoftmaxStrategy());
  registry.register('fused_scale_softmax_forward', new FusedScaleSoftmaxStrategy());
  registry.register('layer_norm_forward', new LayerNormStrategy());
  
  // Register elementwise operation strategies
  const elementwiseStrategy = new ElementwiseStrategy();
  registry.register('scale_forward', elementwiseStrategy);
  registry.register('add_forward', elementwiseStrategy);
  registry.register('relu_forward', elementwiseStrategy);
  
  return registry;
}
