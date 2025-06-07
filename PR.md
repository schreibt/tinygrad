# Add Associative Scan Operation

## Description
This PR adds a general associative scan operation to tinygrad, which is a fundamental operation for parallel prefix computations. The implementation is efficient, parallel, and supports any associative binary operation.

## Motivation
Associative scan (also known as parallel prefix sum) is a crucial operation for many algorithms, including:
- Parallel algorithms for dynamic programming
- Attention mechanisms in transformers
- Parallel reduction operations
- Efficient implementation of cumulative operations

## Implementation Details
The implementation includes:

1. Main method: `associative_scan(op, axis=0, reverse=False)`
   - Takes any associative binary operation
   - Supports any axis
   - Handles reverse scanning
   - Maintains numerical stability

2. Internal methods:
   - `_cumalu`: Handles small tensors (≤32 elements) using a parallel prefix sum algorithm
   - `_split_cumalu`: Handles larger tensors by splitting into segments

3. Key features:
   - Parallel execution for better performance
   - Handles tensors of any size
   - Supports reverse scanning
   - Maintains numerical stability
   - Compatible with tinygrad's lazy evaluation system

## Usage Examples
```python
# Cumulative sum
result = tensor.associative_scan(lambda a, b: a + b, axis=0)

# Cumulative product
result = tensor.associative_scan(lambda a, b: a * b, axis=1)

# Cumulative maximum
result = tensor.associative_scan(lambda a, b: a.maximum(b), axis=0)

# Reverse scan
result = tensor.associative_scan(lambda a, b: a + b, axis=0, reverse=True)
```

## Testing
The implementation includes comprehensive tests that verify:
1. Basic functionality with different tensor sizes
2. Different operations (add, mul, max)
3. Different axes
4. Reverse scanning
5. Gradient computation

## Performance
The implementation is optimized for:
- Small tensors (≤32 elements): Uses a parallel prefix sum algorithm
- Large tensors: Uses a segmented approach to maintain efficiency
- Memory usage: Minimizes temporary allocations
- Parallel execution: Leverages tinygrad's parallel capabilities

## Related Issues
Closes #XXX (Add issue number)

## Checklist
- [x] Implementation
- [x] Tests
- [x] Documentation
- [x] Performance optimization
- [x] Code style compliance 