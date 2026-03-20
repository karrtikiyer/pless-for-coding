# P-Less Sampling: Curated HumanEval Examples

## Executive Summary

- **1** tasks where p-less notably outperforms baselines (advantage > 0.5)
- **54** tasks where baselines notably outperform p-less (advantage < -0.5)
- **109** tasks with comparable performance (|advantage| ≤ 0.5)
- Analyzed across **4 models**: Qwen--Qwen2.5-Coder-7B-Instruct, Qwen--Qwen3-Coder-30B-A3B-Instruct, codellama--CodeLlama-7b-Instruct-hf, mistralai--Codestral-22B-v0.1
- Highest mean diversity ratio: **temp_0.7** (0.443)
- **Qwen--Qwen2.5-Coder-7B-Instruct**: 2 wins, 28 losses, p_less mean=8.33, p_less_norm mean=7.52
- **Qwen--Qwen3-Coder-30B-A3B-Instruct**: 1 wins, 18 losses, p_less mean=7.60, p_less_norm mean=7.56
- **codellama--CodeLlama-7b-Instruct-hf**: 1 wins, 61 losses, p_less mean=3.61, p_less_norm mean=3.59
- **mistralai--Codestral-22B-v0.1**: 3 wins, 33 losses, p_less mean=7.80, p_less_norm mean=7.76

## Methodology

- **Data**: 164 HumanEval tasks × 10 samples × 6 methods × 4 models
- **P-less advantage**: mean across models of max(p_less, p_less_norm) − max(baselines)
- **Baselines**: greedy, temp_0.2, temp_0.7, top_p_0.95
- **Difficulty**: easy (best baseline ≥ 8/10), medium (3–7/10), hard (≤ 2/10)
- Tasks where all methods score 10/10 or 0/10 are excluded from example selection

## Aggregate Statistics

### P-less advantage distribution

| Model | Wins | Losses | Ties |
|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 2 | 28 | 134 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 1 | 18 | 145 |
| codellama--CodeLlama-7b-Instruct-hf | 1 | 61 | 102 |
| mistralai--Codestral-22B-v0.1 | 3 | 33 | 128 |

### By task difficulty

| Difficulty | Qwen--Qwen2.5-Coder-7B-Instruct W/L/T | Qwen--Qwen3-Coder-30B-A3B-Instruct W/L/T | codellama--CodeLlama-7b-Instruct-hf W/L/T | mistralai--Codestral-22B-v0.1 W/L/T |
|---|---|---|---|---|
| Easy | 0/6/69 | 0/1/74 | 1/26/48 | 0/7/68 |
| Medium | 2/20/56 | 1/15/62 | 0/33/45 | 3/20/55 |
| Hard | 0/2/9 | 0/2/9 | 0/2/9 | 0/6/5 |

## Top P-Less Wins (7 examples)

### HumanEval/137

**P-less advantage:** +0.75 | **Difficulty:** medium

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 0 | 1 | 2 | 2 | 0 | 4 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 0 | 0 | 0 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 0 | 2 | 1 | 2 | 3 | 0 |

**Most dramatic model:** Qwen--Qwen2.5-Coder-7B-Instruct — `p_less_norm` (4/10) vs `greedy` (0/10)

<details><summary>P-less (passed) — <code>p_less_norm</code></summary>

```python
    # Convert both values to strings for comparison
    a_str = str(a).replace(',', '.')
    b_str = str(b).replace(',', '.')
    
    # Try converting the strings to floats for numerical comparison
    try:
        a_float = float(a_str)
        b_float = float(b_str)
    except ValueError:
        # If conversion fails, return None
        return None
    
    # Compare the floats
    if a_float > b_float:
        return a
    elif b_float > a_float:
        return b
    else:
        # Return None if values are equal
        return None
```
</details>

<details><summary>Baseline (failed) — <code>greedy</code></summary>

```python
    # Convert both inputs to floats if they are not already
    a = float(a.replace(',', '.'))
    b = float(b.replace(',', '.'))

    # Compare the two values
    if a > b:
        return a
    elif b > a:
        return b
    else:
        return None
```
</details>


### HumanEval/9

**P-less advantage:** +0.50 | **Difficulty:** easy

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 9 | 9 | 10 | 10 | 9 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 5 | 5 | 5 | 7 | 6 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 10 | 10 | 10 | 10 |

**Most dramatic model:** codellama--CodeLlama-7b-Instruct-hf — `p_less` (7/10) vs `greedy` (0/10)

<details><summary>P-less (passed) — <code>p_less</code></summary>

```python
    max_values = []
    current_max = None
    for i, num in enumerate(numbers):
        if current_max is None or num > current_max:
            current_max = num
        max_values.append(current_max)
    return max_values
```
</details>

<details><summary>Baseline (failed) — <code>greedy</code></summary>

```python
    result = []
    for i in range(len(numbers)):
        max_value = numbers[i]
        for j in range(i, len(numbers)):
            if numbers[j] > max_value:
                max_value = numbers[j]
        result.append(max_value)
    return result
```
</details>


### HumanEval/142

**P-less advantage:** +0.50 | **Difficulty:** medium

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 0 | 0 | 3 | 3 | 0 | 3 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 0 | 1 | 4 | 0 | 6 | 5 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 0 | 0 | 0 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 9 | 9 | 10 | 10 |

**Most dramatic model:** Qwen--Qwen3-Coder-30B-A3B-Instruct — `p_less` (6/10) vs `greedy` (0/10)

<details><summary>P-less (passed) — <code>p_less</code></summary>

```python
    if not lst:
        return 0
    
    total_sum = 0
    
    for i in range(len(lst)):
        if i % 3 == 0:
            # Index is multiple of 3, square the value
            total_sum += lst[i] ** 2
        elif i % 4 == 0:
            # Index is multiple of 4 but not multiple of 3, cube the value
            total_sum += lst[i] ** 3
        else:
            # Index is not multiple of 3 or 4, keep the original value
            total_sum += lst[i]
    
    return total_sum
```
</details>

<details><summary>Baseline (failed) — <code>greedy</code></summary>

```python
    if not lst:
        return 0
    
    total = 0
    for i in range(len(lst)):
        if i % 3 ==
```
</details>


### HumanEval/26

**P-less advantage:** +0.25 | **Difficulty:** medium

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 0 | 4 | 9 | 5 | 5 | 3 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 0 | 0 | 0 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 0 | 1 | 1 | 1 | 6 | 3 |

**Most dramatic model:** mistralai--Codestral-22B-v0.1 — `p_less` (6/10) vs `greedy` (0/10)

<details><summary>P-less (passed) — <code>p_less</code></summary>

```python
    count_dict = {}
    for num in numbers:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1

    result = []
    for num in numbers:
        if count_dict[num] == 1:
            result.append(num)

    return result
```
</details>

<details><summary>Baseline (failed) — <code>greedy</code></summary>

```python
    seen = set()
    result = []
    for number in numbers:
        if number not in seen:
            result.append(number)
            seen.add(number)
    return result
```
</details>


### HumanEval/0

**P-less advantage:** +0.00 | **Difficulty:** easy

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 10 | 9 | 9 | 10 | 10 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 10 | 10 | 10 | 10 | 10 | 10 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 10 | 10 | 10 | 10 |

**Most dramatic model:** Qwen--Qwen2.5-Coder-7B-Instruct — `p_less` (10/10) vs `temp_0.7` (9/10)

<details><summary>P-less (passed) — <code>p_less</code></summary>

```python
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```
</details>

<details><summary>Baseline (failed) — <code>temp_0.7</code></summary>

```python
    for num in set(numbers):
        if any(abs(num - other_num) < threshold for other_num in numbers if num != other_num):
            return True
    return False
```
</details>


### HumanEval/4

**P-less advantage:** +0.00 | **Difficulty:** easy

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 10 | 9 | 9 | 10 | 9 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 10 | 10 | 10 | 10 | 10 | 10 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 10 | 10 | 10 | 10 |

**Most dramatic model:** Qwen--Qwen2.5-Coder-7B-Instruct — `p_less` (10/10) vs `temp_0.7` (9/10)

<details><summary>P-less (passed) — <code>p_less</code></summary>

```python
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
```
</details>

<details><summary>Baseline (failed) — <code>temp_0.7</code></summary>

```python
    mean = statistics.mean(numbers)
    return statistics.mean([abs(x - mean) for x in numbers])

# Test the function
print(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]))  # Expected output: 1.0
```
</details>


### HumanEval/7

**P-less advantage:** +0.00 | **Difficulty:** easy

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 10 | 10 | 10 | 10 | 10 | 10 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 10 | 10 | 10 | 10 |

**Most dramatic model:** Qwen--Qwen2.5-Coder-7B-Instruct — `p_less` (10/10) vs `greedy` (10/10)

<details><summary>P-less (passed) — <code>p_less</code></summary>

```python
    return [s for s in strings if substring in s]
```
</details>

<details><summary>Baseline (failed) — <code>greedy</code></summary>

```python
    return [s for s in strings if substring in s]
```
</details>


## Top P-Less Losses (7 examples)

### HumanEval/108

**P-less advantage:** -2.75 | **Difficulty:** medium

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 0 | 4 | 7 | 4 | 3 | 5 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 0 | 2 | 9 | 3 | 0 | 0 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 0 | 0 | 0 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 0 | 0 | 0 | 0 | 0 | 0 |

**Most dramatic model:** Qwen--Qwen3-Coder-30B-A3B-Instruct — `temp_0.7` (9/10) vs `p_less` (0/10)

<details><summary>P-less (failed) — <code>p_less</code></summary>

```python
    Looking at this problem, I need to:

    1. Count elements where the sum of digits is greater than 0
    2. For negative numbers, the first digit is negative (signed digits)
    3. Handle edge cases like empty arrays

    Let me trace through the examples:
    - `count_nums([])` → 0 (empty array)
    - `count_nums([-1, 11, -11])` → 1 
      - -1: digits [-1] → sum = -1 (not > 0)
      - 11: digits [1, 1] → sum = 2 (> 0) ✓
      - -11: digits [-1, 1] → sum = 0 (not > 0)
      - So only 1 element qualifies
    - `count_nums([1, 1, 2])` → 3 (all positive, all sums > 0)

    Wait, let me re-read the example more carefully. For `[-1, 11, -11]`, it should return 1, but my calculation shows only 11 qualifies. Let me check -11 again:
    - -11: digits [-1, 1] → sum = 0 (not > 0)

    Actually, let me re-read... it says "If a number is negative, then its first signed digit will be negative". So for -11, the digits are [-1, 1], sum = 0. 

    But the expected result is 1, so let me check -1:
    - -1: digits [-1] → sum = -1 (not > 0)

    Hmm, let me think differently. Maybe I misunderstood. Let me check if -11 has sum > 0:
    - -11: digits [-1, 1] → sum = 0 (not > 0)

    Actually, let me re-read the problem statement. The example shows `count_nums([-1, 11, -11]) == 1`. Let me check if there's a different interpretation.

    Looking at it again, let me trace each number:
    - -1: digits [-1] → sum = -1 (not > 0)
    - 11: digits [1, 1] → sum = 2 (> 0) ✓
    - -11: digits [-1, 1] → sum = 0 (not > 0)

    Only 11
```
</details>

<details><summary>Baseline (passed) — <code>temp_0.7</code></summary>

```python
    def sum_of_signed_digits(num):
        """Calculate sum of signed digits for a number."""
        if num == 0:
            return 0
        
        # Convert to string to process digits
        str_num = str(abs(num))
        
        # Calculate sum of digits
        digit_sum = sum(int(digit) for digit in str_num)
        
        # If original number was negative, make first digit negative
        if num < 0:
            first_digit = int(str_num[0])
            digit_sum = digit_sum - 2 * first_digit  # Subtract 2*first_digit to make it negative
        
        return digit_sum
    
    count = 0
    for num in arr:
        if sum_of_signed_digits(num) > 0:
            count += 1
    
    return count
```
</details>


### HumanEval/127

**P-less advantage:** -2.50 | **Difficulty:** hard

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 0 | 0 | 0 | 0 | 0 | 0 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 1 | 0 | 2 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 0 | 4 | 5 | 8 | 0 | 0 |

**Most dramatic model:** mistralai--Codestral-22B-v0.1 — `top_p_0.95` (8/10) vs `p_less` (0/10)

<details><summary>P-less (failed) — <code>p_less</code></summary>

```python
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def intersection(interval1, interval2):
    """Check if the length of intersection of two intervals is a prime number."""
    # Calculate the intersection
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])

    # Check if the intervals intersect
    if start > end:
        return "NO"

    # Calculate the length of the intersection
    length = end - start

    # Check if the length is a prime number
    if is_prime(length):
        return "YES"
    else:
        return "NO"
```
</details>

<details><summary>Baseline (passed) — <code>top_p_0.95</code></summary>

```python
    # find the intersection
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])

    # if the intervals don't intersect, return "NO"
    if start > end:
        return "NO"

    # compute the length of the intersection
    length = end - start

    # check if the length is a prime number
    if length > 1:
        for i in range(2, length):
            if (length % i) == 0:
                return "NO"
        else:
            return "YES"
    else:
        return "NO"
```
</details>


### HumanEval/89

**P-less advantage:** -2.50 | **Difficulty:** medium

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 10 | 9 | 10 | 10 | 8 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 0 | 3 | 6 | 5 | 0 | 0 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 0 | 4 | 4 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 10 | 10 | 10 | 10 |

**Most dramatic model:** Qwen--Qwen3-Coder-30B-A3B-Instruct — `temp_0.7` (6/10) vs `p_less` (0/10)

<details><summary>P-less (failed) — <code>p_less</code></summary>

```python
    result = []
    shift = 2 * 2  # shift by two multiplied by two places = 4
    
    for char in s:
        if char.isalpha():
            # Determine if uppercase or lowercase
            if char.isupper():
                # For uppercase letters (A-Z)
                # Convert to 0-25 range, apply shift, wrap around, convert back
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(shifted + ord
```
</details>

<details><summary>Baseline (passed) — <code>temp_0.7</code></summary>

```python
    result = []
    shift = 2 * 2  # shift by 4 positions
    
    for char in s:
        if char.isalpha():
            # Determine if uppercase or lowercase
            if char.isupper():
                # For uppercase letters (A-Z)
                # Convert to 0-25 range, apply shift, wrap around, convert back
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(shifted + ord('A')))
            else:
                # For lowercase letters (a-z)
                # Convert to 0-25 range, apply shift, wrap around, convert back
                shifted = (ord(char) - ord('a') + shift) % 26
                result.append(chr(shifted + ord('a')))
        else:
            # Non-alphabetic characters remain unchanged
            result.append(char)
    
    return ''.join(result)
```
</details>


### HumanEval/88

**P-less advantage:** -2.25 | **Difficulty:** easy

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 9 | 8 | 7 | 7 | 7 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 8 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 10 | 9 | 4 | 4 | 10 | 10 |
| mistralai--Codestral-22B-v0.1 | 0 | 3 | 6 | 6 | 0 | 0 |

**Most dramatic model:** mistralai--Codestral-22B-v0.1 — `temp_0.7` (6/10) vs `p_less` (0/10)

<details><summary>P-less (failed) — <code>p_less</code></summary>

```python
    # Make a copy of the array to avoid modifying the original
    array_copy = array.copy()

    # Check if the sum of the first and last elements is odd or even
    if (array_copy[0] + array_copy[-1]) % 2 == 0:
        # If even, sort the array in descending order
        array_copy.sort(reverse=True)
    else:
        # If odd, sort the array in ascending order
        array_copy.sort()

    return array_copy
```
</details>

<details><summary>Baseline (passed) — <code>temp_0.7</code></summary>

```python
    if not array:
        return []

    first_last_sum = array[0] + array[-1]

    if first_last_sum % 2 == 0:
        return sorted(array, reverse=True)
    else:
        return sorted(array)

# Test Cases
print(sort_array([])) # => []
print(sort_array([5])) # => [5]
print(sort_array([2, 4, 3, 0, 1, 5])) # => [0, 1, 2, 3, 4, 5]
print(sort_array([2, 4, 3, 0, 1, 5, 6])) # => [6, 5, 4, 3, 2, 1, 0]
```
</details>


### HumanEval/71

**P-less advantage:** -2.25 | **Difficulty:** medium

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 0 | 3 | 5 | 2 | 3 | 2 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 0 | 0 | 2 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 10 | 5 | 2 | 5 | 3 | 5 |

**Most dramatic model:** mistralai--Codestral-22B-v0.1 — `greedy` (10/10) vs `p_less` (3/10)

<details><summary>P-less (failed) — <code>p_less</code></summary>

```python
    # Check if the three sides form a valid triangle
    if (a + b > c) and (a + c > b) and (b + c > a):
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        # Calculate the area using Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        # Return the area rounded to 2 decimal points
        return round(area, 2)
    else:
        # Return -1 if the three sides do not form a valid triangle
        return -1
```
</details>

<details><summary>Baseline (passed) — <code>greedy</code></summary>

```python
    # Check if the three sides form a valid triangle
    if (a + b > c) and (a + c > b) and (b + c > a):
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        # Calculate the area using Heron's formula
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
        # Return the area rounded to 2 decimal points
        return round(area, 2)
    else:
        # Return -1 if the three sides do not form a valid triangle
        return -1
```
</details>


### HumanEval/12

**P-less advantage:** -2.25 | **Difficulty:** easy

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 1 | 9 | 6 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 10 | 10 | 10 | 10 |

**Most dramatic model:** codellama--CodeLlama-7b-Instruct-hf — `temp_0.7` (9/10) vs `p_less` (0/10)

<details><summary>P-less (failed) — <code>p_less</code></summary>

```python
    if not strings:
        return None
    longest_string = None
    for string in strings:
        if len(string) > len(longest_string) or longest_string is None:
            longest_string = string
    return longest_string
```
</details>

<details><summary>Baseline (passed) — <code>temp_0.7</code></summary>

```python
    if not strings:
        return None

    longest_string = strings[0]
    for string in strings[1:]:
        if len(string) > len(longest_string):
            longest_string = string

    return longest_string
```
</details>


### HumanEval/6

**P-less advantage:** -2.25 | **Difficulty:** medium

| Model | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Qwen--Qwen2.5-Coder-7B-Instruct | 10 | 10 | 10 | 10 | 10 | 10 |
| Qwen--Qwen3-Coder-30B-A3B-Instruct | 0 | 5 | 9 | 8 | 0 | 0 |
| codellama--CodeLlama-7b-Instruct-hf | 0 | 0 | 0 | 0 | 0 | 0 |
| mistralai--Codestral-22B-v0.1 | 10 | 10 | 9 | 10 | 10 | 10 |

**Most dramatic model:** Qwen--Qwen3-Coder-30B-A3B-Instruct — `temp_0.7` (9/10) vs `p_less` (0/10)

<details><summary>P-less (failed) — <code>p_less</code></summary>

```python
    if not paren_string.strip():
        return []
    
    groups = paren_string.split()
    result = []
    
    for group in groups:
        max_depth = 0
        current_depth = 0
        
        for char in group:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif
```
</details>

<details><summary>Baseline (passed) — <code>temp_0.7</code></summary>

```python
    if not paren_string.strip():
        return []
    
    groups = paren_string.split()
    result = []
    
    for group in groups:
        max_depth = 0
        current_depth = 0
        
        for char in group:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        result.append(max_depth)
    
    return result
```
</details>


## Code Length Analysis

Mean tokens generated per method, grouped by p-less outcome:

| Partition | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Win | 131.5 | 141.0 | 189.1 | 159.6 | 134.6 | 150.5 |
| Loss | 119.7 | 123.4 | 133.3 | 129.6 | 123.2 | 129.0 |
| Tie | 94.8 | 95.1 | 103.5 | 104.4 | 95.0 | 99.7 |

Mean code length (characters):

| Partition (chars) | greedy | temp_0.2 | temp_0.7 | top_p_0.95 | p_less | p_less_norm |
|---|---|---|---|---|---|---|
| Win | 476 | 512 | 655 | 568 | 484 | 546 |
| Loss | 396 | 409 | 438 | 425 | 406 | 426 |
| Tie | 321 | 318 | 341 | 343 | 318 | 334 |

## Diversity Analysis

Mean diversity ratio (num_distinct_correct / num_correct) per method:

| Method | Mean Diversity Ratio | Tasks with ≥1 correct |
|---|---|---|
| greedy | 0.0890 | 445 |
| temp_0.2 | 0.2715 | 504 |
| temp_0.7 | 0.4425 | 550 |
| top_p_0.95 | 0.3825 | 542 |
| p_less | 0.1879 | 478 |
| p_less_norm | 0.2823 | 487 |

## Per-Model Notes

### Qwen--Qwen2.5-Coder-7B-Instruct

- p_less mean score: 8.33/10
- p_less_norm mean score: 7.52/10
- Divergence (|p_less − p_less_norm|): 0.81
- Wins: 2, Losses: 28

### Qwen--Qwen3-Coder-30B-A3B-Instruct

- p_less mean score: 7.60/10
- p_less_norm mean score: 7.56/10
- Divergence (|p_less − p_less_norm|): 0.04
- Wins: 1, Losses: 18

### codellama--CodeLlama-7b-Instruct-hf

- p_less mean score: 3.61/10
- p_less_norm mean score: 3.59/10
- Divergence (|p_less − p_less_norm|): 0.02
- Wins: 1, Losses: 61

### mistralai--Codestral-22B-v0.1

- p_less mean score: 7.80/10
- p_less_norm mean score: 7.76/10
- Divergence (|p_less − p_less_norm|): 0.04
- Wins: 3, Losses: 33

## Appendix: Full Task Ranking

All 164 tasks sorted by p-less advantage:

| Rank | Task ID | P-less Advantage | Difficulty |
|---|---|---|---|
| 1 | HumanEval/137 | +0.75 | medium |
| 2 | HumanEval/9 | +0.50 | easy |
| 3 | HumanEval/142 | +0.50 | medium |
| 4 | HumanEval/26 | +0.25 | medium |
| 5 | HumanEval/0 | +0.00 | easy |
| 6 | HumanEval/4 | +0.00 | easy |
| 7 | HumanEval/7 | +0.00 | easy |
| 8 | HumanEval/13 | +0.00 | easy |
| 9 | HumanEval/14 | +0.00 | easy |
| 10 | HumanEval/15 | +0.00 | easy |
| 11 | HumanEval/16 | +0.00 | easy |
| 12 | HumanEval/17 | +0.00 | medium |
| 13 | HumanEval/18 | +0.00 | easy |
| 14 | HumanEval/20 | +0.00 | easy |
| 15 | HumanEval/21 | +0.00 | easy |
| 16 | HumanEval/22 | +0.00 | easy |
| 17 | HumanEval/23 | +0.00 | easy |
| 18 | HumanEval/25 | +0.00 | easy |
| 19 | HumanEval/27 | +0.00 | easy |
| 20 | HumanEval/28 | +0.00 | easy |
| 21 | HumanEval/29 | +0.00 | easy |
| 22 | HumanEval/30 | +0.00 | easy |
| 23 | HumanEval/31 | +0.00 | easy |
| 24 | HumanEval/34 | +0.00 | easy |
| 25 | HumanEval/35 | +0.00 | easy |
| 26 | HumanEval/42 | +0.00 | easy |
| 27 | HumanEval/43 | +0.00 | easy |
| 28 | HumanEval/44 | +0.00 | easy |
| 29 | HumanEval/45 | +0.00 | easy |
| 30 | HumanEval/47 | +0.00 | easy |
| 31 | HumanEval/48 | +0.00 | easy |
| 32 | HumanEval/53 | +0.00 | easy |
| 33 | HumanEval/56 | +0.00 | easy |
| 34 | HumanEval/58 | +0.00 | easy |
| 35 | HumanEval/60 | +0.00 | easy |
| 36 | HumanEval/61 | +0.00 | easy |
| 37 | HumanEval/64 | +0.00 | medium |
| 38 | HumanEval/66 | +0.00 | easy |
| 39 | HumanEval/68 | +0.00 | easy |
| 40 | HumanEval/72 | +0.00 | easy |
| 41 | HumanEval/74 | +0.00 | easy |
| 42 | HumanEval/78 | +0.00 | medium |
| 43 | HumanEval/80 | +0.00 | medium |
| 44 | HumanEval/84 | +0.00 | medium |
| 45 | HumanEval/86 | +0.00 | easy |
| 46 | HumanEval/92 | +0.00 | easy |
| 47 | HumanEval/96 | +0.00 | medium |
| 48 | HumanEval/98 | +0.00 | medium |
| 49 | HumanEval/100 | +0.00 | medium |
| 50 | HumanEval/104 | +0.00 | easy |
| 51 | HumanEval/109 | +0.00 | medium |
| 52 | HumanEval/112 | +0.00 | easy |
| 53 | HumanEval/116 | +0.00 | medium |
| 54 | HumanEval/125 | +0.00 | medium |
| 55 | HumanEval/130 | +0.00 | hard |
| 56 | HumanEval/131 | +0.00 | medium |
| 57 | HumanEval/132 | +0.00 | hard |
| 58 | HumanEval/135 | +0.00 | medium |
| 59 | HumanEval/136 | +0.00 | easy |
| 60 | HumanEval/139 | +0.00 | medium |
| 61 | HumanEval/141 | +0.00 | medium |
| 62 | HumanEval/145 | +0.00 | hard |
| 63 | HumanEval/146 | +0.00 | medium |
| 64 | HumanEval/149 | +0.00 | easy |
| 65 | HumanEval/150 | +0.00 | easy |
| 66 | HumanEval/152 | +0.00 | easy |
| 67 | HumanEval/158 | +0.00 | medium |
| 68 | HumanEval/162 | +0.00 | hard |
| 69 | HumanEval/163 | +0.00 | hard |
| 70 | HumanEval/1 | -0.25 | medium |
| 71 | HumanEval/2 | -0.25 | medium |
| 72 | HumanEval/5 | -0.25 | medium |
| 73 | HumanEval/11 | -0.25 | easy |
| 74 | HumanEval/37 | -0.25 | medium |
| 75 | HumanEval/38 | -0.25 | medium |
| 76 | HumanEval/39 | -0.25 | medium |
| 77 | HumanEval/59 | -0.25 | easy |
| 78 | HumanEval/65 | -0.25 | medium |
| 79 | HumanEval/67 | -0.25 | medium |
| 80 | HumanEval/70 | -0.25 | medium |
| 81 | HumanEval/73 | -0.25 | medium |
| 82 | HumanEval/77 | -0.25 | medium |
| 83 | HumanEval/91 | -0.25 | medium |
| 84 | HumanEval/101 | -0.25 | medium |
| 85 | HumanEval/105 | -0.25 | medium |
| 86 | HumanEval/117 | -0.25 | medium |
| 87 | HumanEval/122 | -0.25 | medium |
| 88 | HumanEval/133 | -0.25 | hard |
| 89 | HumanEval/148 | -0.25 | medium |
| 90 | HumanEval/153 | -0.25 | medium |
| 91 | HumanEval/19 | -0.50 | medium |
| 92 | HumanEval/41 | -0.50 | medium |
| 93 | HumanEval/50 | -0.50 | easy |
| 94 | HumanEval/51 | -0.50 | easy |
| 95 | HumanEval/63 | -0.50 | medium |
| 96 | HumanEval/69 | -0.50 | medium |
| 97 | HumanEval/85 | -0.50 | easy |
| 98 | HumanEval/97 | -0.50 | easy |
| 99 | HumanEval/102 | -0.50 | medium |
| 100 | HumanEval/110 | -0.50 | medium |
| 101 | HumanEval/118 | -0.50 | medium |
| 102 | HumanEval/121 | -0.50 | easy |
| 103 | HumanEval/129 | -0.50 | medium |
| 104 | HumanEval/134 | -0.50 | medium |
| 105 | HumanEval/147 | -0.50 | medium |
| 106 | HumanEval/151 | -0.50 | easy |
| 107 | HumanEval/156 | -0.50 | easy |
| 108 | HumanEval/157 | -0.50 | easy |
| 109 | HumanEval/160 | -0.50 | hard |
| 110 | HumanEval/161 | -0.50 | easy |
| 111 | HumanEval/3 | -0.75 | easy |
| 112 | HumanEval/8 | -0.75 | easy |
| 113 | HumanEval/33 | -0.75 | medium |
| 114 | HumanEval/46 | -0.75 | easy |
| 115 | HumanEval/75 | -0.75 | medium |
| 116 | HumanEval/95 | -0.75 | medium |
| 117 | HumanEval/106 | -0.75 | medium |
| 118 | HumanEval/107 | -0.75 | easy |
| 119 | HumanEval/113 | -0.75 | easy |
| 120 | HumanEval/128 | -0.75 | medium |
| 121 | HumanEval/143 | -0.75 | easy |
| 122 | HumanEval/155 | -0.75 | easy |
| 123 | HumanEval/159 | -0.75 | medium |
| 124 | HumanEval/32 | -1.00 | hard |
| 125 | HumanEval/36 | -1.00 | medium |
| 126 | HumanEval/49 | -1.00 | easy |
| 127 | HumanEval/52 | -1.00 | easy |
| 128 | HumanEval/54 | -1.00 | medium |
| 129 | HumanEval/55 | -1.00 | medium |
| 130 | HumanEval/79 | -1.00 | easy |
| 131 | HumanEval/94 | -1.00 | medium |
| 132 | HumanEval/111 | -1.00 | medium |
| 133 | HumanEval/114 | -1.00 | easy |
| 134 | HumanEval/123 | -1.00 | easy |
| 135 | HumanEval/124 | -1.00 | medium |
| 136 | HumanEval/10 | -1.25 | medium |
| 137 | HumanEval/24 | -1.25 | easy |
| 138 | HumanEval/81 | -1.25 | medium |
| 139 | HumanEval/82 | -1.25 | easy |
| 140 | HumanEval/119 | -1.25 | medium |
| 141 | HumanEval/140 | -1.25 | medium |
| 142 | HumanEval/154 | -1.25 | medium |
| 143 | HumanEval/57 | -1.50 | medium |
| 144 | HumanEval/76 | -1.50 | medium |
| 145 | HumanEval/87 | -1.50 | medium |
| 146 | HumanEval/90 | -1.50 | easy |
| 147 | HumanEval/99 | -1.50 | easy |
| 148 | HumanEval/103 | -1.50 | medium |
| 149 | HumanEval/115 | -1.50 | hard |
| 150 | HumanEval/120 | -1.50 | hard |
| 151 | HumanEval/138 | -1.50 | medium |
| 152 | HumanEval/40 | -1.75 | easy |
| 153 | HumanEval/62 | -1.75 | easy |
| 154 | HumanEval/144 | -1.75 | medium |
| 155 | HumanEval/83 | -2.00 | medium |
| 156 | HumanEval/93 | -2.00 | medium |
| 157 | HumanEval/126 | -2.00 | medium |
| 158 | HumanEval/6 | -2.25 | medium |
| 159 | HumanEval/12 | -2.25 | easy |
| 160 | HumanEval/71 | -2.25 | medium |
| 161 | HumanEval/88 | -2.25 | easy |
| 162 | HumanEval/89 | -2.50 | medium |
| 163 | HumanEval/127 | -2.50 | hard |
| 164 | HumanEval/108 | -2.75 | medium |
