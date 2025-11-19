# Documentation Style Guide

## File Header Template

```python
"""
<Module/Script Name>

<One-line description>

<Detailed description paragraph explaining purpose, key concepts, and usage context>

Authors: <Names>
Date: <Creation Date>
Last Modified: <Date>

Acknowledgements:
- Boston Dynamics Spot SDK examples
- <Other sources/inspirations>

License: <License info>
"""
```

## Module Docstrings

```python
"""
<Module Name>

<Overview paragraph>

Components:
- <Component 1>: <Description>
- <Component 2>: <Description>

Usage:
    from src.module import Component
    
    component = Component(...)
    result = component.method()

Dependencies:
- <Dependency 1>
- <Dependency 2>
"""
```

## Class Docstrings

```python
class ClassName:
    """
    <One-line description>
    
    <Detailed description of class purpose and behavior>
    
    Attributes:
        attr1 (type): Description
        attr2 (type): Description
    
    Usage:
        obj = ClassName(arg1, arg2)
        result = obj.method()
    
    Notes:
        - Important note 1
        - Important note 2
    """
```

## Method/Function Docstrings

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    <One-line description>
    
    <Detailed description if needed>
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this exception occurs
    
    Notes:
        - Implementation detail 1
        - Implementation detail 2
    
    Example:
        >>> result = function_name(val1, val2)
        >>> print(result)
        <expected output>
    """
```

## Inline Comments

- Use for complex logic or non-obvious code
- Explain **why**, not **what** (code shows what)
- Keep comments close to relevant code
- Update comments when code changes

Example:
```python
# Compute person position in body frame using spherical coordinates
# Pan/tilt from PTZ represent spherical angles in body frame
person_x = distance_m * math.cos(tilt_rad) * math.cos(pan_rad)
```

## Section Headers (for long files)

```python
# ============================================================================
# Section Name
# ============================================================================
```

## TODOs and FIXMEs

```python
# TODO(author): Description of what needs to be done
# FIXME(author): Description of bug/issue
# HACK(author): Description of workaround and why needed
```

## Consistency Guidelines

1. **Tense**: Use present tense ("Returns" not "Will return")
2. **Voice**: Use imperative mood for first line ("Compute" not "Computes")
3. **Clarity**: Be specific about types, units, coordinate frames
4. **Examples**: Include for non-trivial usage
5. **Cross-references**: Link to related functions/classes when helpful

## Coordinate Frame Documentation

Always specify coordinate frames for spatial data:

```python
def transform_point(point: np.ndarray, frame_name: str) -> np.ndarray:
    """
    Transform point from body frame to vision frame.
    
    Args:
        point: 3D point in body frame (x forward, y left, z up)
        frame_name: Target frame name ("vision", "odom", etc.)
    
    Returns:
        3D point in specified frame
    """
```

## Units Documentation

Always specify units for measurements:

```python
def compute_distance(pose: SE3Pose) -> float:
    """
    Compute distance to target.
    
    Returns:
        Distance in meters
    """
```

## Boston Dynamics SDK References

When using SDK patterns, cite the example:

```python
# Pattern from fetch.py example in Spot SDK
# https://github.com/boston-dynamics/spot-sdk/python/examples/fetch
```
