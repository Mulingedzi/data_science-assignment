# Reflection

## Which method worked better and why?
Collaborative filtering produced good recommendations because users with similar tastes rated similar movies.  
Content-based also worked since genres (Sci-Fi, Action, etc.) were included.

## Challenges
- Handling missing ratings (NaN values)
- Avoiding recommending movies the user already watched
- Balancing between user-user and genre-based recommendations

## Next steps
- Combine collaborative and content-based (hybrid)
- Try item-item similarity
- Evaluate accuracy with precision/recall
