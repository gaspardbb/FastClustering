# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gaspardb/Documents/stage_mit/code/clustering

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gaspardb/Documents/stage_mit/code/clustering/build

# Include any dependencies generated for this target.
include CMakeFiles/clustering_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/clustering_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/clustering_test.dir/flags.make

CMakeFiles/clustering_test.dir/tests/main.cpp.o: CMakeFiles/clustering_test.dir/flags.make
CMakeFiles/clustering_test.dir/tests/main.cpp.o: ../tests/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaspardb/Documents/stage_mit/code/clustering/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/clustering_test.dir/tests/main.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/clustering_test.dir/tests/main.cpp.o -c /home/gaspardb/Documents/stage_mit/code/clustering/tests/main.cpp

CMakeFiles/clustering_test.dir/tests/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/clustering_test.dir/tests/main.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaspardb/Documents/stage_mit/code/clustering/tests/main.cpp > CMakeFiles/clustering_test.dir/tests/main.cpp.i

CMakeFiles/clustering_test.dir/tests/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/clustering_test.dir/tests/main.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaspardb/Documents/stage_mit/code/clustering/tests/main.cpp -o CMakeFiles/clustering_test.dir/tests/main.cpp.s

CMakeFiles/clustering_test.dir/tests/main.cpp.o.requires:

.PHONY : CMakeFiles/clustering_test.dir/tests/main.cpp.o.requires

CMakeFiles/clustering_test.dir/tests/main.cpp.o.provides: CMakeFiles/clustering_test.dir/tests/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/clustering_test.dir/build.make CMakeFiles/clustering_test.dir/tests/main.cpp.o.provides.build
.PHONY : CMakeFiles/clustering_test.dir/tests/main.cpp.o.provides

CMakeFiles/clustering_test.dir/tests/main.cpp.o.provides.build: CMakeFiles/clustering_test.dir/tests/main.cpp.o


CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o: CMakeFiles/clustering_test.dir/flags.make
CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o: ../lib/googletest/googletest/src/gtest_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaspardb/Documents/stage_mit/code/clustering/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o -c /home/gaspardb/Documents/stage_mit/code/clustering/lib/googletest/googletest/src/gtest_main.cc

CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaspardb/Documents/stage_mit/code/clustering/lib/googletest/googletest/src/gtest_main.cc > CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.i

CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaspardb/Documents/stage_mit/code/clustering/lib/googletest/googletest/src/gtest_main.cc -o CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.s

CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.requires:

.PHONY : CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.requires

CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.provides: CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.requires
	$(MAKE) -f CMakeFiles/clustering_test.dir/build.make CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.provides.build
.PHONY : CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.provides

CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.provides.build: CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o


# Object files for target clustering_test
clustering_test_OBJECTS = \
"CMakeFiles/clustering_test.dir/tests/main.cpp.o" \
"CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o"

# External object files for target clustering_test
clustering_test_EXTERNAL_OBJECTS =

clustering_test: CMakeFiles/clustering_test.dir/tests/main.cpp.o
clustering_test: CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o
clustering_test: CMakeFiles/clustering_test.dir/build.make
clustering_test: lib/libgtestd.a
clustering_test: CMakeFiles/clustering_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gaspardb/Documents/stage_mit/code/clustering/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable clustering_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/clustering_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/clustering_test.dir/build: clustering_test

.PHONY : CMakeFiles/clustering_test.dir/build

CMakeFiles/clustering_test.dir/requires: CMakeFiles/clustering_test.dir/tests/main.cpp.o.requires
CMakeFiles/clustering_test.dir/requires: CMakeFiles/clustering_test.dir/lib/googletest/googletest/src/gtest_main.cc.o.requires

.PHONY : CMakeFiles/clustering_test.dir/requires

CMakeFiles/clustering_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clustering_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clustering_test.dir/clean

CMakeFiles/clustering_test.dir/depend:
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gaspardb/Documents/stage_mit/code/clustering /home/gaspardb/Documents/stage_mit/code/clustering /home/gaspardb/Documents/stage_mit/code/clustering/build /home/gaspardb/Documents/stage_mit/code/clustering/build /home/gaspardb/Documents/stage_mit/code/clustering/build/CMakeFiles/clustering_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/clustering_test.dir/depend

