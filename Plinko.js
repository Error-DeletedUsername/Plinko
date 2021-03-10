class Plinko{
    constructor(){
        options={
            isStatic: true,
            restitution: 1.0,
            density: 1.0,
        }
        this.radius = 10
        this.body = Bodies.circle(x,y,this.radius,options)
        World.add(world, this.body)
    }
    display(){
        var pos = this.body.position
        var angle = this.body.angle

        push()
        translate(pos.x, pos.y)
        rotate(angle)
        imageMode(CENTER)
        noStroke()
        fill("white")
        ellispeMode(RADIUS)
        ellispe(0,0,this.radius,this.r)
    }
}