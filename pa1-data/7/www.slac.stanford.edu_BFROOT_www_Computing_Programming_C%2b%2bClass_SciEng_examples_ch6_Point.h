example programs from the book scientific and engineering programming in c++ an introduction with advanced techniques and examples addison wesley 1994 c copyright international business machines corporation 1994 all rights reserved see readme file for further details include iostream h include math h typedef float number class line class point public point create uninitialized point number x number y create from x y number distance point point const distance to another point number distance line line const distance to a line number& x reference to x coordinate number x const get x coordinate number& y reference to y coordinate number y const get y coordinate number angle point p1 point p3 const private number the_x x coordinate number the_y y coordinate inline point point inline point point number x number y the_x x the_y y inline number& point x return the_x inline number& point y return the_y inline number point x const return the_x inline number point y const return the_y inline istream& operator istream& is point& p return is px py inline ostream& operator ostream& os const point& p return os px py inline number point angle point p1 point p3 const number v21 2 p1 x x p1 y y number v23 2 p3 x x p3 y y number dot_product v21 0 v23 0 v21 1 v23 1 number cross_product v23 0 v21 1 v23 1 v21 0 number ang atan2 cross_product dot_product if ang 0 ang 2.0 m_pi return ang
