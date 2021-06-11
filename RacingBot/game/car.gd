extends Sprite

export var maxSpeed = 1
export var maxTurn = 0.5
export var acceleration = 0.2
export var deacceleration = 0.1
export var brake = 1.2

onready var travelLength = null
onready var speed = 0
onready var currentSpeed = 0
onready var speedDesired = 0
onready var turn = 0
onready var turnStrength = 1
onready var rayL = $RayCast_left
onready var rayR = $RayCast_right
onready var LRayDistance
onready var RRayDistance
onready var LRayNormal
onready var RRayNormal
onready var controller = get_tree().get_root().get_node("track").get_node("controller")
onready var pathFollow = get_tree().get_root().get_node("track").get_node("Path2D/PathFollow2D")
onready var path = get_tree().get_root().get_node("track").get_node("Path2D")

func _process(_delta):
	if rotation_degrees >= 360:
		rotation_degrees -= 360
	elif rotation_degrees < 0:
		rotation_degrees += 360

	# movement
	if Input.is_action_pressed("ui_up") && (speed < maxSpeed):
		speed += acceleration
	if Input.is_action_pressed("ui_down") && (speed > 0):
		speed -= brake

	if speed > 0:
		speed -= deacceleration
	else:
		speed = 0

	# TODO make turnStrength exponential? pow((speed / maxSpeed), 0.5)

	if Input.is_action_pressed("ui_left") && speed > 0:
		rotation_degrees -= maxTurn * turnStrength
	if Input.is_action_pressed("ui_right") && speed > 0:
		rotation_degrees += maxTurn * turnStrength
	if Input.is_action_pressed("ui_accept"):
		checkRayCast()
	position -= Vector2(0, speed).rotated(deg2rad(rotation_degrees))
	#print(output())

func _on_Area2D_area_entered(area):
	if area is Wall:
		travelLength = path.curve.get_closest_offset(Vector2(position)) / path.curve.get_baked_length()
		print("collision at distance: %f" % travelLength)
		# stop the car
		maxSpeed = 0 
		speed = 0

func checkRayCast():
	if rayL.is_colliding():
		LRayDistance = (rayL.get_collision_point() - position).length()
		LRayNormal = rad2deg(rayL.get_collision_normal().rotated(PI/2).angle()) - rotation_degrees + 180
		# relative to car rotation

		#print("collision normal: %f" % (rad2deg(rayL.get_collision_normal().angle()) + 180))
		#print("rotation degrees: %f" % rotation_degrees)
		print(LRayNormal)
	else:
		LRayDistance = 99999
		LRayNormal = 0
	if rayR.is_colliding():
		RRayDistance = (rayR.get_collision_point() - position).length()
		RRayNormal = rad2deg(rayR.get_collision_normal().rotated(PI/2).angle()) - rotation_degrees + 180
	else:
		RRayDistance = 99999
		RRayNormal = 0
func output():
	checkRayCast()
	if maxSpeed != 0:
		currentSpeed = speed / maxSpeed
	else:
		currentSpeed = 0
	return [currentSpeed, log(LRayDistance), log(RRayDistance)]

func input(speedIn, turnIn):
	if speedIn < -0.3:
		speedDesired = -1
	elif speedIn > 0.3:
		speedDesired = 1
	else:
		speedDesired = 0

	turn = turnIn


