#include <CGAL/Simple_cartesian.h>
#include <CGAL/convex_hull_2.h>
#include <vector>
#include <iostream>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_2 Point;

int main()
{
    std::vector<Point> points = {Point(0, 0), Point(10, 0), Point(5, 5), Point(3, 2)};
    std::vector<Point> result;

    CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(result));

    std::cout << "Convex Hull:\n";
    for (auto &p : result)
        std::cout << p << "\n";
    return 0;
}
