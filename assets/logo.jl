using Luxor



Drawing(600, 600, "hello-world.png")
origin()

sethue(30/100, 43/100, 96/100)
rad = 200
circle(0, 0, rad, :fill)


sethue("red")

cent_ellipse = Point(0, 0)
ellipse(cent_ellipse, 2*rad, rad/10, :stroke)

pole = Point(0, -rad)
arrow(cent_ellipse, pole, linewidth = 2.0)

x = Point(rad, 0)
arrow(cent_ellipse, x, linewidth = 2.0)

θ = rand() * 2π
input_state = Point(rad * cos(θ), rad * sin(θ) )
line(cent_ellipse, input_state, :stroke)
circle(input_state, rad/20, :fill)



θ = rand() * 2π
final_state = Point(-rad * cos(θ), rad * sin(θ) )
line(cent_ellipse, final_state, :stroke)
circle(final_state, rad/20, :fill)


# random state between the two other ones



curve(input_state, x, final_state)
sethue("black")
setdash("shortdashed")
# arrow(input_state, final_state, [15, 15])
strokepath()

finish()
preview()