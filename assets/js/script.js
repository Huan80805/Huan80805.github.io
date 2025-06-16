'use strict';

// element toggle function
const elementToggleFunc = function (elem) { elem.classList.toggle("active"); }

// custom select variables for project filtering
const select = document.querySelector("[data-select]");
const selectItems = document.querySelectorAll("[data-select-item]");
const selectValue = document.querySelector("[data-selecct-value]");
const filterBtn = document.querySelectorAll("[data-filter-btn]");

if(select) {
  select.addEventListener("click", function () { elementToggleFunc(this); });

  // add event in all select items
  for (let i = 0; i < selectItems.length; i++) {
    selectItems[i].addEventListener("click", function () {

      let selectedValue = this.innerText.toLowerCase();
      selectValue.innerText = this.innerText;
      elementToggleFunc(select);
      filterFunc(selectedValue);

    });
  }
}


// filter variables
const filterItems = document.querySelectorAll("[data-filter-item]");

const filterFunc = function (selectedValue) {

  for (let i = 0; i < filterItems.length; i++) {

    if (selectedValue === "all") {
      filterItems[i].classList.add("active");
    } else if (selectedValue === filterItems[i].dataset.category) {
      filterItems[i].classList.add("active");
    } else {
      filterItems[i].classList.remove("active");
    }

  }

}

// add event in all filter button items for large screen
let lastClickedBtn = filterBtn[0];

for (let i = 0; i < filterBtn.length; i++) {

  filterBtn[i].addEventListener("click", function () {

    let selectedValue = this.innerText.toLowerCase();
    selectValue.innerText = this.innerText;
    filterFunc(selectedValue);

    if (lastClickedBtn) {
      lastClickedBtn.classList.remove("active");
    }
    this.classList.add("active");
    lastClickedBtn = this;

  });
}

// contact form variables (如果未來要加回來可以保留)
// const form = document.querySelector("[data-form]");
// const formInputs = document.querySelectorAll("[data-form-input]");
// const formBtn = document.querySelector("[data-form-btn]");

// for (let i = 0; i < formInputs.length; i++) {
//   formInputs[i].addEventListener("input", function () {
//     if (form.checkValidity()) {
//       formBtn.removeAttribute("disabled");
//     } else {
//       formBtn.setAttribute("disabled", "");
//     }
//   });
// }


// New Page Navigation Logic for Scrolling
const navigationLinks = document.querySelectorAll("[data-nav-link]");
const sections = document.querySelectorAll("[data-page]");

// Function to handle scroll events and update active nav link
const scrollHandler = () => {
  const scrollPosition = window.scrollY;

  sections.forEach(section => {
    // 減去一個偏移量 (150px) 讓標題進入畫面中間時才觸發
    if (scrollPosition >= section.offsetTop - 150 && scrollPosition < section.offsetTop + section.offsetHeight - 150) {
      const currentId = section.id;
      
      navigationLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentId}`) {
          link.classList.add('active');
        }
      });
    }
  });
};

// Add scroll event listener
window.addEventListener('scroll', scrollHandler);

// Add click event to all nav links to handle smooth scroll (if CSS fails) and active state
navigationLinks.forEach(link => {
  link.addEventListener('click', (event) => {
    // In case html scroll-behavior is not supported
    // event.preventDefault();
    // const targetId = link.getAttribute('href');
    // document.querySelector(targetId).scrollIntoView({
    //   behavior: 'smooth'
    // });

    // Instantly set active class on click
    navigationLinks.forEach(nav => nav.classList.remove('active'));
    link.classList.add('active');
  });
});