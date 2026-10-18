add_executable(metal_onthefly_bench run.cpp)
target_link_libraries(metal_onthefly_bench PUBLIC besthea besthea_metal)
install(TARGETS metal_onthefly_bench
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME})
