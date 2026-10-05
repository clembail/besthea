add_executable(metal_onthefly_test run.cpp)
target_link_libraries(metal_onthefly_test PUBLIC besthea besthea_metal)
install(TARGETS metal_onthefly_test
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
