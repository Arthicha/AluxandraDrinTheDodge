treesMax = 28; %How many multiple trees (must be at least 2, 1 for source and 1 for destination
seedsPerAxis = 3; %Number of seeds allowed on each axis (discretely placed seeds which idealy helps the RRT expansion)
wallCount = 5; %Number of mock walls to be placed in the environment
rrt = RrtPlanner(treesMax,seedsPerAxis,wallCount);
rrt.SetStart([0 -0.9 0]);
rrt.SetGoal([0 +0.9 0]);
rrt.smoothedPath
% plot3(rrt.smoothedPath(:,1),rrt.smoothedPath(:,2),rrt.smoothedPath(:,3),'k*');
rrt.Run()
delete(rrt);

obstacleFilename = 'obstacles.txt';
seedsPerAxis = 7;
treesMax = seedsPerAxis^3*3+2;
rrt = RrtPlanner(treesMax,seedsPerAxis,obstacleFilename);
rrt.drawingSkipsPerDrawing = 30;
rrt = RrtPlanner(treesMax,seedsPerAxis,obstacleFilename);
rrt.Run()
plot3(rrt.path(:,1),rrt.path(:,2),rrt.path(:,3),'k*');
