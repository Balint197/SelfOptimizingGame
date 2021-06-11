extends Sprite

export var maxTurn = 5
export var maxSpeed = 20
export var acceleration = 1
export var deacceleration = 0.3
export var brake = 1.2

onready var deactivated = 0
onready var time = 0
onready var finished = 0
onready var travelLength = 0
onready var speed = 0
onready var currentSpeed = 0
onready var speedDesired = 0
onready var turn = 0
onready var turnStrength = 1
onready var rayL = $RayCast_left
onready var rayL2 = $RayCast_left2
onready var rayR = $RayCast_right
onready var rayR2 = $RayCast_right2
onready var rayM = $RayCast_mid
onready var LRayDistance
onready var RRayDistance
onready var LRay2Distance
onready var RRay2Distance
onready var MRayDistance
onready var LRayNormal
onready var RRayNormal
onready var controller = get_tree().get_root().get_node("track").get_node("controller")
onready var pathFollow = get_tree().get_root().get_node("track").get_node("Path2D/PathFollow2D")
onready var path = get_tree().get_root().get_node("track").get_node("Path2D")

func _process(_delta):
	if deactivated == 0:
		if rotation_degrees >= 360:
			rotation_degrees -= 360
		elif rotation_degrees < 0:
			rotation_degrees += 360
			
	#	input([speedController.value,turnController.value])
		# movement
		if speedDesired == 1 && (speed < maxSpeed):
			speed += acceleration
		elif speedDesired == -1 && (speed > 0):
			speed -= brake
		
		if speed > 0:
			speed -= deacceleration
		else:
			speed = 0
		# TODO make turnStrength exponential? pow((speed / maxSpeed), 0.5)
		if speed > 0:
			rotation_degrees += maxTurn * turn

		position -= Vector2(0, speed).rotated(deg2rad(rotation_degrees))

func _on_Area2D_area_entered(area):
	if area is Finish:
		getTravelLength()
		#print("collision at distance: %f" % travelLength)
		# stop the car
		maxSpeed = 0 
		speed = 0
		finished = 1

	if area is Wall:
		getTravelLength()
		#print("collision at distance: %f" % travelLength)
		# stop the car
		maxSpeed = 0
		speed = 0

func getTravelLength():
	var travelLength = path.curve.get_closest_offset(Vector2(position)) / path.curve.get_baked_length()
	return travelLength

func checkRayCast():
	if rayL.is_colliding():
		LRayDistance = (rayL.get_collision_point() - position).length()
		#LRayNormal = rad2deg(rayL.get_collision_normal().rotated(PI/2).angle()) - rotation_degrees + 180
		# relative to car rotation
	else:
		LRayDistance = 99999
		#LRayNormal = 0
	if rayL2.is_colliding():
		LRay2Distance = (rayL2.get_collision_point() - position).length()
		#LRayNormal = rad2deg(rayL.get_collision_normal().rotated(PI/2).angle()) - rotation_degrees + 180
		# relative to car rotation
	else:
		LRay2Distance = 99999
		#LRayNormal = 0
	if rayR.is_colliding():
		RRayDistance = (rayR.get_collision_point() - position).length()
		#RRayNormal = rad2deg(rayR.get_collision_normal().rotated(PI/2).angle()) - rotation_degrees + 180
	else:
		RRayDistance = 99999
		#RRayNormal = 0
	if rayR2.is_colliding():
		RRay2Distance = (rayR2.get_collision_point() - position).length()
		#RRayNormal = rad2deg(rayR.get_collision_normal().rotated(PI/2).angle()) - rotation_degrees + 180
	else:
		RRay2Distance = 99999
		#RRayNormal = 0
	if rayM.is_colliding():
		MRayDistance = (rayM.get_collision_point() - position).length()
		#RRayNormal = rad2deg(rayR.get_collision_normal().rotated(PI/2).angle()) - rotation_degrees + 180
	else:
		MRayDistance = 99999
		#RRayNormal = 0
func output():
	if deactivated == 0:
		checkRayCast()
	if maxSpeed != 0:
		currentSpeed = speed / maxSpeed
	else:
		currentSpeed = 0
	return [currentSpeed, log(LRayDistance), log(RRayDistance), log(LRay2Distance), log(RRay2Distance), log(MRayDistance)]

func input(input):					# input = [speed, turn]
	if input[0] < -0.3:
		speedDesired = -1
	elif input[0] > 0.3:
		speedDesired = 1
	else:
		speedDesired = 0

	turn = input[1]
	
func checkDistance():
	var distance = get_tree().get_root().get_node("track").get_node("Path2D").curve.get_closest_offset(position)
	
func quitSim():
	get_tree().quit()
