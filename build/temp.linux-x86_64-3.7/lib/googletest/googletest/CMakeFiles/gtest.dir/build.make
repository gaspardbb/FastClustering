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
CMAKE_BINARY_DIR = /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7

# Include any dependencies generated for this target.
include lib/googletest/googletest/CMakeFiles/gtest.dir/depend.make

# Include the progress variables for this target.
include lib/googletest/googletest/CMakeFiles/gtest.dir/progress.make

# Include the compile flags for this target's objects.
include lib/googletest/googletest/CMakeFiles/gtest.dir/flags.make

lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o: lib/googletest/googletest/CMakeFiles/gtest.dir/flags.make
lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o: ../../lib/googletest/googletest/src/gtest-all.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o"
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gtest.dir/src/gtest-all.cc.o -c /home/gaspardb/Documents/stage_mit/code/clustering/lib/googletest/googletest/src/gtest-all.cc

lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gtest.dir/src/gtest-all.cc.i"
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaspardb/Documents/stage_mit/code/clustering/lib/googletest/googletest/src/gtest-all.cc > CMakeFiles/gtest.dir/src/gtest-all.cc.i

lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gtest.dir/src/gtest-all.cc.s"
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaspardb/Documents/stage_mit/code/clustering/lib/googletest/googletest/src/gtest-all.cc -o CMakeFiles/gtest.dir/src/gtest-all.cc.s

lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires:

.PHONY : lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires

lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides: lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires
	$(MAKE) -f lib/googletest/googletest/CMakeFiles/gtest.dir/build.make lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides.build
.PHONY : lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides

lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides.build: lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o


# Object files for target gtest
gtest_OBJECTS = \
"CMakeFiles/gtest.dir/src/gtest-all.cc.o"

# External object files for target gtest
gtest_EXTERNAL_OBJECTS =

lib/libgtest.a: lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
lib/libgtest.a: lib/googletest/googletest/CMakeFiles/gtest.dir/build.make
lib/libgtest.a: lib/googletest/googletest/CMakeFiles/gtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../libgtest.a"
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest && $(CMAKE_COMMAND) -P CMakeFiles/gtest.dir/cmake_clean_target.cmake
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/googletest/googletest/CMakeFiles/gtest.dir/build: lib/libgtest.a

.PHONY : lib/googletest/googletest/CMakeFiles/gtest.dir/build

lib/googletest/googletest/CMakeFiles/gtest.dir/requires: lib/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires

.PHONY : lib/googletest/googletest/CMakeFiles/gtest.dir/requires

lib/googletest/googletest/CMakeFiles/gtest.dir/clean:
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest && $(CMAKE_COMMAND) -P CMakeFiles/gtest.dir/cmake_clean.cmake
.PHONY : lib/googletest/googletest/CMakeFiles/gtest.dir/clean

lib/googletest/googletest/CMakeFiles/gtest.dir/depend:
	cd /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gaspardb/Documents/stage_mit/code/clustering /home/gaspardb/Documents/stage_mit/code/clustering/lib/googletest/googletest /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7 /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest /home/gaspardb/Documents/stage_mit/code/clustering/build/temp.linux-x86_64-3.7/lib/googletest/googletest/CMakeFiles/gtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/googletest/googletest/CMakeFiles/gtest.dir/depend

